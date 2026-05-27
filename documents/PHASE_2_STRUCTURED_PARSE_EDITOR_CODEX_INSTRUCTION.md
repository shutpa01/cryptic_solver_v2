# Phase 2 Structured Parse Editor — Codex Implementation Instruction
# Revision 2 — 2026-05-27

---

## Purpose

Add a container-specific structured parse editor that:

1. Stores a JSON-based structured parse (authoritative) in a new
   `manual_structured_parses` table.
2. Merges the parse into the display using a new merge function that correctly
   produces split answer-tile colouring for container clues.
3. Provides a human-friendly container form in the admin UI (no graph concepts).
4. Demotes the existing graph editor to a legacy section.

The SWALLOW acceptance test (see section 9) is the primary deliverable. Do not
ship without passing all checks in section 10.

---

## Design principles (do not deviate)

- **Pieces own answer tile colours. Operations validate combinations.**
  For a container clue, the outer piece's letters colour their answer boxes
  (including the non-contiguous suffix) even though the outer piece feeds the
  container operation. The operation block shows in the display but never claims
  answer tiles.
- **JSON is authoritative.** The `manual_structured_parses` table holds the parse.
  The existing graph tables are legacy; they continue to work for old data but
  receive no new writes from the structured parse path.
- **One structured parse per clue.** Write replaces the previous row for the
  same `clue_id`. There is no version history in this slice.
- **Answer boxes are 1-based** everywhere: in the UI, in the stored JSON, and in
  the validation logic. Convert to 0-based only at the point of indexing into a
  Python string (i.e. `answer[box - 1]`).
- **Word positions are 0-based** internally and in stored JSON. In the container
  form, the user does NOT enter word positions. The server derives them by
  searching the clue text for the entered phrase.

---

## Files to modify

1. `signature_solver/manual_evidence_store.py` — append new code at end of file
2. `web/routes/clue.py` — modify the admin merge section only
3. `web/routes/admin.py` — add helper + new routes; update 4 existing routes
4. `web/templates/clue.html` — add container form; demote old forms to legacy
5. `web/templates/partials/manual_evidence_nodes.html` — add structured parse display

Do NOT touch any stage engine, any automatic solver, any proof-attempt table, or
any file not listed above.

---

## 1. manual_evidence_store.py — new additions

**Two kinds of change in this file:**

- Sections 1.1 and 1.2 (constants and DDL): append to the bottom of the file.
- Section 1.3 (`ensure_tables`): **edit the existing function in place** — do not
  append a second copy. Find the existing `ensure_tables` definition and replace
  its body with the version below.
- Sections 1.4 onwards: append to the bottom of the file.

### 1.1 Colour mapping constant

```python
# Named colour to CSS role string. Matches piece_N classes in atomic_parse.html.
_COLOUR_TO_ROLE = {
    "blue": "piece_0",
    "pink": "piece_1",
    "yellow": "piece_2",
    "orange": "piece_3",
    "purple": "piece_4",
}
```

### 1.2 DDL for new table

```python
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
```

### 1.3 Update ensure_tables

Replace the body of `ensure_tables` so that the structured parse table is also
created. The new body must call BOTH the existing DDL and the new DDL:

```python
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
```

### 1.4 write_structured_parse

```python
def write_structured_parse(clue_id, parse_dict, source="human",
                           status="draft", conn=None):
    """Save or replace the structured parse for a clue. Returns the row id.

    If a row already exists for clue_id it is updated in place (one parse per
    clue). Otherwise a new row is inserted.
    """
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
```

### 1.5 get_structured_parse

```python
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
```

### 1.6 delete_structured_parse

```python
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
```

### 1.7 _validate_structured_parse

```python
def _validate_structured_parse(parse_dict, answer):
    """Validate a structured parse dict against a clue answer.

    Returns a list of error strings. Empty list means valid.
    Answer boxes in parse_dict are 1-based.
    """
    errors = []
    cleaned = _clean_answer(answer)
    if not cleaned:
        errors.append("answer is empty after cleaning")
        return errors

    # Definition
    defn = parse_dict.get("definition") or {}
    if not defn:
        errors.append("missing definition block")
    elif not defn.get("clue_text"):
        errors.append("definition has no clue_text")

    # Pieces
    pieces = parse_dict.get("pieces") or []
    piece_ids = set()
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
        mapping = piece.get("mapping", "positional")
        colour = (piece.get("colour") or "").lower()

        if colour and colour not in _COLOUR_TO_ROLE:
            errors.append(
                "piece %s: unknown colour %r (valid: %s)"
                % (pid, colour, ", ".join(sorted(_COLOUR_TO_ROLE)))
            )

        if mapping == "positional":
            if letters and boxes:
                if len(letters) != len(boxes):
                    errors.append(
                        "piece %s: %d boxes but %d letters"
                        % (pid, len(boxes), len(letters))
                    )
                else:
                    for i, box in enumerate(boxes):
                        idx = box - 1  # 1-based → 0-based
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

    # Operations
    operations = parse_dict.get("operations") or []
    for op in operations:
        oid = op.get("id", "?")
        op_type = (op.get("type") or "").lower()

        if op_type == "container":
            outer_id = op.get("outer_piece_id")
            inner_id = op.get("inner_piece_id")

            if not outer_id or outer_id not in piece_ids:
                errors.append(
                    "operation %s: outer_piece_id %r not in pieces" % (oid, outer_id)
                )
            if not inner_id or inner_id not in piece_ids:
                errors.append(
                    "operation %s: inner_piece_id %r not in pieces" % (oid, inner_id)
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
                    elif inner_letters:
                        # Test every occurrence of inner_letters; accept if
                        # any removal leaves exactly outer_letters.
                        valid_insertion = False
                        search_start = 0
                        while True:
                            idx = result.find(inner_letters, search_start)
                            if idx == -1:
                                break
                            remainder = (
                                result[:idx] + result[idx + len(inner_letters):]
                            )
                            if remainder == outer_letters:
                                valid_insertion = True
                                break
                            search_start = idx + 1
                        if not valid_insertion:
                            errors.append(
                                "operation %s: no valid position exists where "
                                "removing inner %r from result %r leaves "
                                "outer %r"
                                % (oid, inner_letters, result, outer_letters)
                            )
                    if result != cleaned:
                        errors.append(
                            "operation %s: result %r does not match cleaned "
                            "answer %r" % (oid, result, cleaned)
                        )

    return errors
```

### 1.8 _find_word_positions

Note: this function is intentionally simple for this slice. It handles the
common case well enough (space/comma separated words, case-insensitive). It
will not handle all punctuation edge cases. That is acceptable because
click-to-fill is deferred and positions are never entered by the user in the
primary container form. Do not attempt to make it more robust here.

```python
def _find_word_positions(phrase, clue_text):
    """Return list of 0-based word indices of phrase in clue_text.

    Tokenises both strings on whitespace and common punctuation. Returns []
    if phrase is not found. Case-insensitive.
    """
    if not phrase or not clue_text:
        return []
    sep = re.compile(r"[\s,;:]+")
    clue_words = [w for w in sep.split(clue_text.strip()) if w]
    phrase_words = [w for w in sep.split(phrase.strip()) if w]
    if not phrase_words:
        return []
    n = len(phrase_words)
    for i in range(len(clue_words) - n + 1):
        if [w.lower() for w in clue_words[i:i + n]] == [
            w.lower() for w in phrase_words
        ]:
            return list(range(i, i + n))
    return []
```

### 1.9 merge_structured_parse_into_display

```python
def merge_structured_parse_into_display(wfw_display, parse_dict, answer):
    """Merge a structured parse into an existing wfw_display dict.

    Core rule: pieces own answer tile colours; operations validate combinations.
    A piece's answer_boxes are always honoured even when the piece feeds an
    operation. The operation block is rendered as an OP_BLOCK but claims no
    answer tiles.

    answer_boxes in parse_dict are 1-based. Internally converted to 0-based
    for indexing.
    """
    cleaned = _clean_answer(answer)

    # --- evaluate operations (for OP_BLOCK status) ---
    pieces_by_id = {p["id"]: p for p in (parse_dict.get("pieces") or [])}
    op_results = {}  # op_id -> (ok, reason)

    for op in (parse_dict.get("operations") or []):
        oid = op.get("id", "?")
        op_type = (op.get("type") or "").lower()

        if op_type == "container":
            outer_id = op.get("outer_piece_id")
            inner_id = op.get("inner_piece_id")
            outer = pieces_by_id.get(outer_id)
            inner = pieces_by_id.get(inner_id)
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
            else:
                # Test every occurrence of inner_letters; accept if any
                # removal leaves exactly outer_letters.
                valid_insertion = False
                search_start = 0
                while True:
                    idx = result.find(inner_letters, search_start)
                    if idx == -1:
                        break
                    remainder = result[:idx] + result[idx + len(inner_letters):]
                    if remainder == outer_letters:
                        valid_insertion = True
                        break
                    search_start = idx + 1
                if not valid_insertion:
                    op_results[oid] = (
                        False,
                        "no valid position where removing inner %r from "
                        "result %r leaves outer %r"
                        % (inner_letters, result, outer_letters),
                    )
                elif result != cleaned:
                    op_results[oid] = (
                        False,
                        "result %r does not match answer %r" % (result, cleaned),
                    )
                else:
                    op_results[oid] = (True, "")
        else:
            # Other operation types: mark as unverified but not failed
            op_results[oid] = (True, "")

    # --- build clue blocks ---
    new_blocks = []
    covered_word_positions = set()

    # Definition block
    defn = parse_dict.get("definition") or {}
    if defn.get("clue_text"):
        positions = defn.get("clue_word_positions") or []
        covered_word_positions.update(positions)
        span = (
            [min(positions), max(positions) + 1] if positions else [0, 1]
        )
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

    # Piece blocks (SOURCE_BLOCK)
    for piece in (parse_dict.get("pieces") or []):
        positions = piece.get("clue_word_positions") or []
        covered_word_positions.update(positions)
        span = (
            [min(positions), max(positions) + 1] if positions else [0, 1]
        )
        letters = (piece.get("letters") or "").upper()
        boxes = piece.get("answer_boxes") or []
        mapping = piece.get("mapping", "positional")
        colour = (piece.get("colour") or "").lower()
        role = _COLOUR_TO_ROLE.get(colour, "synonym")

        # Determine validity for block status
        valid = True
        reason = None
        if mapping == "positional" and letters and boxes:
            if len(letters) != len(boxes):
                valid = False
                reason = "box count %d != letter count %d" % (
                    len(boxes), len(letters))
            else:
                for i, box in enumerate(boxes):
                    idx = box - 1  # 1-based → 0-based
                    if idx < 0 or idx >= len(cleaned):
                        valid = False
                        reason = "box %d out of range" % box
                        break
                    if letters[i] != cleaned[idx]:
                        valid = False
                        reason = "letter %r at box %d != answer %r" % (
                            letters[i], box, cleaned[idx])
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

    # Operation blocks (OP_BLOCK)
    for op in (parse_dict.get("operations") or []):
        positions = op.get("clue_word_positions") or []
        covered_word_positions.update(positions)
        span = (
            [min(positions), max(positions) + 1] if positions else [0, 1]
        )
        op_type = (op.get("type") or "").lower()
        ok, reason = op_results.get(op.get("id", "?"), (False, "not evaluated"))
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

    # Filler blocks
    for filler in (parse_dict.get("filler") or []):
        positions = filler.get("clue_word_positions") or []
        covered_word_positions.update(positions)
        span = (
            [min(positions), max(positions) + 1] if positions else [0, 1]
        )
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

    # Retain existing auto-generated blocks for uncovered clue positions
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

    # --- build answer_links ---
    # CRITICAL: pieces own answer tile colours. The container operation does
    # not claim any tiles. Pieces assign their boxes regardless of whether
    # they feed an operation.
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
        span = (
            [min(positions), max(positions) + 1] if positions else None
        )

        if mapping == "positional":
            if not letters or not boxes or len(letters) != len(boxes):
                continue
            for i, box in enumerate(boxes):
                idx = box - 1  # 1-based → 0-based
                if idx < 0 or idx >= len(cleaned):
                    continue
                if letters[i] != cleaned[idx]:
                    continue  # failed letter check; leave tile plain
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
            # Anagram fodder: colour all claimed boxes without letter check
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

    # Fall back to auto links for uncovered positions
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
```

---

## 2. web/routes/clue.py — modify admin merge section

Find the block at approximately lines 452–489 that reads:

```python
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
                clue_dict["wfw_display"] = {
                    "status": "wfw_review",
                    "blocks": [],
                    "answer_links": [],
                    "clue_text": clue["clue_text"] or "",
                    "answer": clue["answer"] or "",
                    "tokens": [],
                    "operations": [],
                    "review_messages": [
                        "Manual evidence only - no automatic proof"
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

Replace it entirely with:

```python
    _manual_nodes = []
    _manual_edges = []
    _manual_structured_parse = None
    if g.get("is_admin", False):
        try:
            from signature_solver.manual_evidence_store import (
                get_graph_for_clue as _get_manual_graph,
                get_structured_parse as _get_structured_parse,
                merge_manual_into_display as _merge_manual_into_display,
                merge_structured_parse_into_display as _merge_sp_into_display,
            )
            _manual_structured_parse = _get_structured_parse(clue_id)
            _graph = _get_manual_graph(clue_id)
            _manual_nodes = _graph["nodes"]
            _manual_edges = _graph["edges"]
        except Exception:
            _manual_nodes = []
            _manual_edges = []
            _manual_structured_parse = None

        if _manual_structured_parse is not None:
            # Structured parse path: JSON is authoritative, graph is legacy.
            if clue_dict.get("wfw_display") is None:
                clue_dict["wfw_display"] = {
                    "status": "wfw_review",
                    "blocks": [],
                    "answer_links": [],
                    "clue_text": clue["clue_text"] or "",
                    "answer": clue["answer"] or "",
                    "tokens": [],
                    "operations": [],
                    "review_messages": [
                        "Manual structured parse — no automatic proof"
                    ],
                    "objections": [],
                    "proof_source": "manual_only",
                    "proof_row_id": None,
                    "missing_enrichments": [],
                }
            _merge_sp_into_display(
                clue_dict["wfw_display"],
                _manual_structured_parse,
                clue["answer"] or "",
            )
        elif _manual_nodes:
            # Legacy graph path (no structured parse exists for this clue).
            if clue_dict.get("wfw_display") is None:
                clue_dict["wfw_display"] = {
                    "status": "wfw_review",
                    "blocks": [],
                    "answer_links": [],
                    "clue_text": clue["clue_text"] or "",
                    "answer": clue["answer"] or "",
                    "tokens": [],
                    "operations": [],
                    "review_messages": [
                        "Manual evidence only - no automatic proof"
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
    clue_dict["manual_structured_parse"] = _manual_structured_parse
```

---

## 3. web/routes/admin.py — new helper + new routes + update existing routes

### 3.1 Add private helper after existing imports/top of file

Add this private function anywhere before the first route function in the file:

```python
def _render_evidence_partial(clue_id, db):
    """Return the manual_evidence_nodes partial with all evidence for a clue.

    Fetches both legacy graph data and the structured parse (if any) so that
    the partial always has current data regardless of which path wrote it.
    """
    from signature_solver.manual_evidence_store import (
        get_edges_for_clue,
        get_nodes_for_clue,
        get_structured_parse,
    )
    nodes = get_nodes_for_clue(clue_id, conn=db)
    edges = get_edges_for_clue(clue_id, conn=db)
    structured_parse = get_structured_parse(clue_id, conn=db)
    return render_template(
        "partials/manual_evidence_nodes.html",
        clue_id=clue_id,
        nodes=nodes,
        edges=edges,
        structured_parse=structured_parse,
    )
```

### 3.2 Update four existing routes to use _render_evidence_partial

In each of the four routes listed below, replace the block that calls
`get_nodes_for_clue`, `get_edges_for_clue`, and `render_template(...)` with a
single call to `_render_evidence_partial(clue_id, db)`.

Routes to update:
- `create_manual_evidence_node` (POST /manual-evidence/<clue_id>)
- `create_manual_evidence_edge` (POST /manual-evidence/<clue_id>/edge)
- `delete_manual_evidence_edge` (POST /manual-evidence/edge/<edge_id>/delete)
- `delete_manual_evidence_node` (POST /manual-evidence/node/<node_id>/delete)

Example: the end of `create_manual_evidence_node` currently reads:

```python
    nodes = get_nodes_for_clue(clue_id, conn=db)
    edges = get_edges_for_clue(clue_id, conn=db)
    return render_template(
        "partials/manual_evidence_nodes.html",
        clue_id=clue_id,
        nodes=nodes,
        edges=edges,
    )
```

Replace with:

```python
    return _render_evidence_partial(clue_id, db)
```

Remove the now-unused local imports of `get_nodes_for_clue` and
`get_edges_for_clue` from those routes (but only if `write_node`, `write_edge`,
`delete_node`, or `delete_edge` are still needed; keep those imports).

### 3.3 Add save_container_structured_parse route

```python
@bp.route("/structured-parse/<int:clue_id>/container", methods=["POST"])
def save_container_structured_parse(clue_id):
    """Save a container structured parse submitted from the container form."""
    _require_admin()
    db = get_admin_db()
    clue = db.execute(
        "SELECT id, clue_text, answer FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    if clue is None:
        abort(404)

    answer = (clue["answer"] or "").strip().upper()
    clue_text = (clue["clue_text"] or "").strip()

    def_text = (request.form.get("def_text") or "").strip()
    p1_text = (request.form.get("p1_text") or "").strip()
    p1_relationship = (request.form.get("p1_relationship") or "synonym").strip()
    p1_letters = (request.form.get("p1_letters") or "").strip().upper()
    p1_boxes_raw = (request.form.get("p1_answer_boxes") or "").strip()
    p1_colour = (request.form.get("p1_colour") or "blue").strip().lower()
    p2_text = (request.form.get("p2_text") or "").strip()
    p2_relationship = (request.form.get("p2_relationship") or "synonym").strip()
    p2_letters = (request.form.get("p2_letters") or "").strip().upper()
    p2_boxes_raw = (request.form.get("p2_answer_boxes") or "").strip()
    p2_colour = (request.form.get("p2_colour") or "pink").strip().lower()
    op_text = (request.form.get("op_text") or "").strip()
    op_outer = (request.form.get("op_outer") or "piece1").strip()

    def parse_boxes(raw):
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except (ValueError, TypeError):
            return []

    p1_boxes = parse_boxes(p1_boxes_raw)
    p2_boxes = parse_boxes(p2_boxes_raw)

    from signature_solver.manual_evidence_store import (
        _find_word_positions,
        _validate_structured_parse,
        write_structured_parse,
    )

    def_positions = _find_word_positions(def_text, clue_text)
    p1_positions = _find_word_positions(p1_text, clue_text)
    p2_positions = _find_word_positions(p2_text, clue_text)
    op_positions = _find_word_positions(op_text, clue_text)

    outer_piece_id = "piece1" if op_outer == "piece1" else "piece2"
    inner_piece_id = "piece2" if op_outer == "piece1" else "piece1"

    parse_dict = {
        "version": 1,
        "clue_id": clue_id,
        "answer": answer,
        "source": "human",
        "confidence": "verified",
        "created_by": "admin",
        "definition": {
            "id": "def1",
            "clue_text": def_text,
            "clue_word_positions": def_positions,
            "answer": answer,
        },
        "pieces": [
            {
                "id": "piece1",
                "clue_text": p1_text,
                "clue_word_positions": p1_positions,
                "relationship": p1_relationship,
                "letters": p1_letters,
                "answer_boxes": p1_boxes,
                "mapping": "positional",
                "colour": p1_colour,
            },
            {
                "id": "piece2",
                "clue_text": p2_text,
                "clue_word_positions": p2_positions,
                "relationship": p2_relationship,
                "letters": p2_letters,
                "answer_boxes": p2_boxes,
                "mapping": "positional",
                "colour": p2_colour,
            },
        ],
        "transform_pieces": [],
        "operations": [
            {
                "id": "op1",
                "clue_text": op_text,
                "clue_word_positions": op_positions,
                "type": "container",
                "outer_piece_id": outer_piece_id,
                "inner_piece_id": inner_piece_id,
                "result": answer,
            }
        ],
        "filler": [],
    }

    errors = _validate_structured_parse(parse_dict, answer)
    if errors:
        error_html = (
            '<div id="manual-evidence-list-%d">'
            '<p class="text-xs font-semibold text-red-600 mt-1">'
            "Parse failed validation:</p>"
            '<ul class="text-xs text-red-600 list-disc ml-4 mt-1">'
            % clue_id
        )
        for err in errors:
            error_html += "<li>%s</li>" % _html_escape(err)
        error_html += "</ul></div>"
        from flask import make_response
        return make_response(error_html, 200)

    write_structured_parse(clue_id, parse_dict, source="human",
                           status="verified", conn=db)
    db.commit()
    return _render_evidence_partial(clue_id, db)
```

### 3.4 Add delete_structured_parse_route route

```python
@bp.route("/structured-parse/<int:clue_id>/delete", methods=["POST"])
def delete_structured_parse_route(clue_id):
    """Delete the structured parse for a clue (not the graph evidence)."""
    _require_admin()
    db = get_admin_db()
    clue = db.execute(
        "SELECT id FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    if clue is None:
        abort(404)

    from signature_solver.manual_evidence_store import delete_structured_parse
    delete_structured_parse(clue_id, conn=db)
    db.commit()
    return _render_evidence_partial(clue_id, db)
```

---

## 4. web/templates/clue.html — modify admin editor section

Find the `<details>` block beginning at approximately line 377:

```html
    <details class="mt-3">
        <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-700">Admin: manual parse editor</summary>
        <div class="mt-2 rounded border border-violet-200 bg-violet-50 px-4 py-3 space-y-4">
```

Replace the entire contents of the `<div class="mt-2 rounded border ...">` (but
not the outer `<details>` tags themselves) with the following. The container
form is primary. The legacy graph editor forms are demoted to a nested
`<details>` at the bottom.

The `manual_evidence_nodes.html` partial include remains where it currently is,
but also receives `structured_parse=clue.manual_structured_parse`.

```html
        <div class="mt-2 rounded border border-violet-200 bg-violet-50 px-4 py-3 space-y-4">

            {# === CONTAINER PARSE FORM (primary) === #}
            <form hx-post="/admin/structured-parse/{{ clue.id }}/container"
                  hx-target="#manual-evidence-list-{{ clue.id }}"
                  hx-swap="outerHTML"
                  hx-on::after-request="this.reset()"
                  class="grid grid-cols-1 gap-3 text-sm rounded border border-teal-200 bg-white/80 p-3">
                <p class="text-xs font-semibold uppercase text-teal-700">Container parse editor</p>
                <p class="text-xs text-slate-500">Enter clue phrases exactly as they appear. Answer boxes are 1-based (first letter = 1).</p>

                <div class="grid grid-cols-1 gap-1">
                    <p class="text-xs font-semibold text-green-700">Definition</p>
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Clue words
                        <input name="def_text" placeholder="e.g. Bird" required
                               class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                    </label>
                </div>

                <div class="grid grid-cols-1 gap-1">
                    <p class="text-xs font-semibold text-sky-700">Piece 1 (outer)</p>
                    <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Clue words
                            <input name="p1_text" placeholder="e.g. female" required
                                   class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                        </label>
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Relationship
                            <select name="p1_relationship"
                                    class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                <option value="synonym">synonym</option>
                                <option value="abbreviation">abbreviation</option>
                                <option value="literal_letters">literal letters</option>
                                <option value="initial_letters">initial letters</option>
                                <option value="foreign">foreign word</option>
                                <option value="pronoun">pronoun/name</option>
                                <option value="single_letter">single letter</option>
                                <option value="hidden_letters">hidden letters</option>
                            </select>
                        </label>
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Letters produced
                            <input name="p1_letters" placeholder="SOW"
                                   class="rounded border border-slate-300 px-2 py-1 bg-white uppercase font-mono text-xs">
                        </label>
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Answer boxes (1-based)
                            <input name="p1_answer_boxes" placeholder="1,6,7"
                                   class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                        </label>
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Colour
                            <select name="p1_colour"
                                    class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                <option value="blue">blue</option>
                                <option value="pink">pink</option>
                                <option value="yellow">yellow</option>
                                <option value="orange">orange</option>
                                <option value="purple">purple</option>
                            </select>
                        </label>
                    </div>
                </div>

                <div class="grid grid-cols-1 gap-1">
                    <p class="text-xs font-semibold text-pink-700">Piece 2 (inner)</p>
                    <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Clue words
                            <input name="p2_text" placeholder="e.g. fence" required
                                   class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                        </label>
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Relationship
                            <select name="p2_relationship"
                                    class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                <option value="synonym">synonym</option>
                                <option value="abbreviation">abbreviation</option>
                                <option value="literal_letters">literal letters</option>
                                <option value="initial_letters">initial letters</option>
                                <option value="foreign">foreign word</option>
                                <option value="pronoun">pronoun/name</option>
                                <option value="single_letter">single letter</option>
                                <option value="hidden_letters">hidden letters</option>
                            </select>
                        </label>
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Letters produced
                            <input name="p2_letters" placeholder="WALL"
                                   class="rounded border border-slate-300 px-2 py-1 bg-white uppercase font-mono text-xs">
                        </label>
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Answer boxes (1-based)
                            <input name="p2_answer_boxes" placeholder="2,3,4,5"
                                   class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                        </label>
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Colour
                            <select name="p2_colour"
                                    class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                <option value="pink" selected>pink</option>
                                <option value="blue">blue</option>
                                <option value="yellow">yellow</option>
                                <option value="orange">orange</option>
                                <option value="purple">purple</option>
                            </select>
                        </label>
                    </div>
                </div>

                <div class="grid grid-cols-1 gap-1">
                    <p class="text-xs font-semibold text-amber-700">Container indicator</p>
                    <div class="grid grid-cols-2 md:grid-cols-3 gap-2">
                        <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            Indicator words
                            <input name="op_text" placeholder="e.g. going over"
                                   class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                        </label>
                        <fieldset class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                            <legend class="text-xs font-semibold text-slate-700">Outer piece</legend>
                            <label class="flex items-center gap-1 font-normal">
                                <input type="radio" name="op_outer" value="piece1" checked> Piece 1
                            </label>
                            <label class="flex items-center gap-1 font-normal">
                                <input type="radio" name="op_outer" value="piece2"> Piece 2
                            </label>
                        </fieldset>
                    </div>
                </div>

                <button type="submit"
                        class="text-xs px-3 py-1.5 rounded border border-teal-500 bg-teal-50 text-teal-800 hover:bg-teal-100 cursor-pointer font-semibold w-fit">
                    Save container parse
                </button>
            </form>

            {# Evidence list (HTMX target).
               Use {% with %} to pass clue.manual_structured_parse explicitly —
               "with context" does not expose nested attributes as bare names. #}
            {% with structured_parse=clue.manual_structured_parse %}
                {% include "partials/manual_evidence_nodes.html" %}
            {% endwith %}

            {# === LEGACY GRAPH EDITOR (advanced / fallback) === #}
            <details class="mt-2">
                <summary class="text-xs text-slate-400 cursor-pointer hover:text-slate-600">Legacy graph editor (advanced)</summary>
                <div class="mt-2 space-y-3">

                    <form hx-post="/admin/manual-evidence/{{ clue.id }}"
                          hx-target="#manual-evidence-list-{{ clue.id }}"
                          hx-swap="outerHTML"
                          hx-on::after-request="this.reset()"
                          class="grid grid-cols-1 gap-2 text-sm rounded border border-sky-100 bg-white/80 p-3">
                        <input type="hidden" name="node_type" value="source">
                        <p class="text-xs font-semibold uppercase text-sky-700">Clue words give letters</p>
                        <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Clue words
                                <input name="word_text" placeholder="CAT" required
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Word numbers (0-based)
                                <input name="word_indices" placeholder="0 or 0,1" required
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Relationship
                                <select name="role"
                                        class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                    <option value="synonym">synonym</option>
                                    <option value="abbreviation">abbreviation</option>
                                    <option value="literal_source">literal letters</option>
                                    <option value="pronoun">pronoun/name</option>
                                    <option value="foreign">foreign word</option>
                                </select>
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Gives
                                <input name="raw_letters" placeholder="TOM"
                                       class="rounded border border-slate-300 px-2 py-1 bg-white uppercase font-mono text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Answer positions (0-based)
                                <input name="answer_positions" placeholder="3,4"
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                        </div>
                        <div class="flex flex-wrap items-end gap-2">
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Colour
                                <select name="group_id"
                                        class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                    <option value="">none</option>
                                    <option value="0">blue</option>
                                    <option value="1">pink</option>
                                    <option value="2">yellow</option>
                                    <option value="3">orange</option>
                                    <option value="4">purple</option>
                                </select>
                            </label>
                            <button type="submit"
                                    class="text-xs px-3 py-1 rounded border border-sky-400 text-sky-700 hover:bg-sky-50 cursor-pointer">
                                Save clue letters
                            </button>
                        </div>
                    </form>

                    <form hx-post="/admin/manual-evidence/{{ clue.id }}"
                          hx-target="#manual-evidence-list-{{ clue.id }}"
                          hx-swap="outerHTML"
                          hx-on::after-request="this.reset()"
                          class="grid grid-cols-1 gap-2 text-sm rounded border border-amber-100 bg-white/80 p-3">
                        <input type="hidden" name="node_type" value="operator">
                        <p class="text-xs font-semibold uppercase text-amber-700">Clue words apply an operation</p>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Indicator words
                                <input name="word_text" placeholder="BACKS" required
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Word numbers (0-based)
                                <input name="word_indices" placeholder="1" required
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Operation
                                <select name="role"
                                        class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                    <option value="reversal">reverse</option>
                                    <option value="deletion">delete letters</option>
                                    <option value="anagram">rearrange</option>
                                    <option value="container">put inside</option>
                                </select>
                            </label>
                            <div class="flex items-end">
                                <button type="submit"
                                        class="text-xs px-3 py-1 rounded border border-amber-400 text-amber-700 hover:bg-amber-50 cursor-pointer">
                                    Save operation
                                </button>
                            </div>
                        </div>
                    </form>

                    <form hx-post="/admin/manual-evidence/{{ clue.id }}"
                          hx-target="#manual-evidence-list-{{ clue.id }}"
                          hx-swap="outerHTML"
                          hx-on::after-request="this.reset()"
                          class="grid grid-cols-1 gap-2 text-sm rounded border border-violet-100 bg-white/80 p-3">
                        <input type="hidden" name="node_type" value="transform">
                        <p class="text-xs font-semibold uppercase text-violet-700">Operation result goes in the answer</p>
                        <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Result label
                                <input name="word_text" placeholder="MOT" required
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Result letters
                                <input name="raw_letters" placeholder="MOT"
                                       class="rounded border border-slate-300 px-2 py-1 bg-white uppercase font-mono text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Answer positions (0-based)
                                <input name="answer_positions" placeholder="0,1,2"
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Colour
                                <select name="group_id"
                                        class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                    <option value="">none</option>
                                    <option value="0">blue</option>
                                    <option value="1">pink</option>
                                    <option value="2">yellow</option>
                                    <option value="3">orange</option>
                                    <option value="4">purple</option>
                                </select>
                            </label>
                            <div class="flex items-end">
                                <button type="submit"
                                        class="text-xs px-3 py-1 rounded border border-violet-400 text-violet-700 hover:bg-violet-50 cursor-pointer">
                                    Save result
                                </button>
                            </div>
                        </div>
                        <details class="text-xs text-slate-500">
                            <summary class="cursor-pointer">Extra operation details</summary>
                            <textarea name="payload_json" rows="2" placeholder='{"delete_text": "ED"}'
                                      class="mt-1 w-full rounded border border-slate-300 px-2 py-1 bg-white font-mono text-xs"></textarea>
                        </details>
                    </form>

                    <form hx-post="/admin/manual-evidence/{{ clue.id }}"
                          hx-target="#manual-evidence-list-{{ clue.id }}"
                          hx-swap="outerHTML"
                          hx-on::after-request="this.reset()"
                          class="grid grid-cols-1 gap-2 text-sm rounded border border-green-100 bg-white/80 p-3">
                        <p class="text-xs font-semibold uppercase text-green-700">Mark definition or filler</p>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Clue words
                                <input name="word_text" placeholder="feature of cricket" required
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Word numbers (0-based)
                                <input name="word_indices" placeholder="4,5,6" required
                                       class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                            </label>
                            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                Mark as
                                <select name="node_type"
                                        class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                    <option value="definition">definition</option>
                                    <option value="structural">link/filler/surface</option>
                                </select>
                            </label>
                            <div class="flex items-end">
                                <button type="submit"
                                        class="text-xs px-3 py-1 rounded border border-green-400 text-green-700 hover:bg-green-50 cursor-pointer">
                                    Save mark
                                </button>
                            </div>
                        </div>
                    </form>

                    {% set connectable = namespace(parts=0, operations=0) %}
                    {% for node in clue.manual_evidence_nodes %}
                        {% if node.node_type in ['source', 'transform'] %}
                            {% set connectable.parts = connectable.parts + 1 %}
                        {% elif node.node_type == 'operator' %}
                            {% set connectable.operations = connectable.operations + 1 %}
                        {% endif %}
                    {% endfor %}
                    <div class="grid grid-cols-1 gap-2 text-sm rounded border border-slate-200 bg-white/80 p-3">
                        <p class="text-xs font-semibold uppercase text-slate-700">Connect clue letters, operation, and result</p>
                        {% if connectable.parts and connectable.operations %}
                        <form hx-post="/admin/manual-evidence/{{ clue.id }}/edge"
                              hx-target="#manual-evidence-list-{{ clue.id }}"
                              hx-swap="outerHTML"
                              hx-on::after-request="this.reset()"
                              class="grid grid-cols-1 gap-2">
                            <div class="grid grid-cols-1 md:grid-cols-[1fr_1fr_11rem] gap-2">
                                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                    This part
                                    <select name="from_node_id"
                                            class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                        {% for node in clue.manual_evidence_nodes %}
                                        {% if node.node_type in ['source', 'transform'] %}
                                        <option value="{{ node.id }}">#{{ node.id }} {{ node.word_text }}{% if node.raw_letters %} -> {{ node.raw_letters }}{% endif %}</option>
                                        {% endif %}
                                        {% endfor %}
                                    </select>
                                </label>
                                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                    To this operation
                                    <select name="to_node_id"
                                            class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                        {% for node in clue.manual_evidence_nodes %}
                                        {% if node.node_type == 'operator' %}
                                        <option value="{{ node.id }}">#{{ node.id }} {{ node.word_text }}</option>
                                        {% endif %}
                                        {% endfor %}
                                    </select>
                                </label>
                                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                                    Connection type
                                    <select name="edge_type"
                                            class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                                        <option value="input_to">feeds into</option>
                                        <option value="output_of">is result of</option>
                                    </select>
                                </label>
                            </div>
                            <button type="submit"
                                    class="text-xs px-3 py-1 rounded border border-slate-400 text-slate-700 hover:bg-slate-50 cursor-pointer w-fit">
                                Save connection
                            </button>
                        </form>
                        {% else %}
                        <p class="text-xs text-slate-400">Save at least one clue-letters entry and one operation to connect them.</p>
                        {% endif %}
                    </div>

                </div>
            </details>

        </div>
```

**Important**: the `{% include "partials/manual_evidence_nodes.html" with context %}`
line that was previously at the bottom of this section should now appear
BETWEEN the container form and the legacy details section (as shown above). The
include must pass `clue.manual_structured_parse` — because the template uses
`with context` this variable is available automatically via `clue.manual_structured_parse`.

---

## 5. web/templates/partials/manual_evidence_nodes.html — add structured parse display

The partial receives an optional `structured_parse` variable. When it is defined
and not None/empty, show the structured parse in crossword-native wording as the
primary display. When it is absent, fall back to the existing graph node list.

Replace the entire file with the following:

```html
{# Manual evidence list.
   Variables:
     clue_id (int)
     nodes (list of dicts) — legacy graph nodes
     edges (list of dicts) — legacy graph edges
     structured_parse (dict or None) — structured parse JSON if present #}
<div id="manual-evidence-list-{{ clue_id }}">
{% set edges = edges if edges is defined else [] %}
{% set structured_parse = structured_parse if structured_parse is defined else none %}

{% if structured_parse %}

{# === STRUCTURED PARSE DISPLAY === #}
<p class="text-xs font-semibold text-teal-700 mt-1 mb-1">Saved structured parse</p>
<div class="rounded border border-teal-100 bg-white px-3 py-2 space-y-1 text-xs">

    {% if structured_parse.definition and structured_parse.definition.clue_text %}
    <div class="flex items-center gap-2">
        <span class="rounded bg-green-100 px-1.5 py-0.5 font-semibold text-green-700 shrink-0">definition</span>
        <span class="font-medium text-gray-900">{{ structured_parse.definition.clue_text }}</span>
        <span class="text-gray-400">=</span>
        <span class="font-mono text-emerald-700">{{ structured_parse.answer }}</span>
    </div>
    {% endif %}

    {% for piece in (structured_parse.pieces or []) %}
    <div class="flex items-center gap-2">
        {% if piece.colour == 'blue' %}
        <span class="rounded bg-sky-100 px-1.5 py-0.5 font-semibold text-sky-700 shrink-0">{{ piece.colour }}</span>
        {% elif piece.colour == 'pink' %}
        <span class="rounded bg-pink-100 px-1.5 py-0.5 font-semibold text-pink-700 shrink-0">{{ piece.colour }}</span>
        {% elif piece.colour == 'yellow' %}
        <span class="rounded bg-yellow-100 px-1.5 py-0.5 font-semibold text-yellow-700 shrink-0">{{ piece.colour }}</span>
        {% elif piece.colour == 'orange' %}
        <span class="rounded bg-orange-100 px-1.5 py-0.5 font-semibold text-orange-700 shrink-0">{{ piece.colour }}</span>
        {% elif piece.colour == 'purple' %}
        <span class="rounded bg-purple-100 px-1.5 py-0.5 font-semibold text-purple-700 shrink-0">{{ piece.colour }}</span>
        {% else %}
        <span class="rounded bg-slate-100 px-1.5 py-0.5 font-semibold text-slate-600 shrink-0">piece</span>
        {% endif %}
        <span class="font-medium text-gray-900">{{ piece.clue_text }}</span>
        <span class="text-gray-400">
            {% if piece.relationship == 'synonym' %}synonym:
            {% elif piece.relationship == 'abbreviation' %}abbrev:
            {% elif piece.relationship == 'literal_letters' %}literal:
            {% elif piece.relationship == 'initial_letters' %}initials:
            {% elif piece.relationship == 'foreign' %}foreign:
            {% elif piece.relationship == 'pronoun' %}name:
            {% elif piece.relationship == 'single_letter' %}letter:
            {% elif piece.relationship == 'hidden_letters' %}hidden:
            {% else %}gives:{% endif %}
        </span>
        <span class="font-mono text-emerald-700">{{ piece.letters or '?' }}</span>
        {% if piece.answer_boxes %}
        <span class="text-gray-400">, boxes</span>
        <span class="font-mono text-gray-700">{{ piece.answer_boxes | join(',') }}</span>
        {% endif %}
    </div>
    {% endfor %}

    {% for op in (structured_parse.operations or []) %}
    {% set op_ok = true %}
    <div class="flex items-center gap-2">
        <span class="rounded bg-amber-100 px-1.5 py-0.5 font-semibold text-amber-700 shrink-0">{{ op.type or 'operation' }}</span>
        <span class="font-medium text-gray-900">{{ op.clue_text }}</span>
        <span class="text-gray-400">:</span>
        {% if op.type == 'container' %}
            {# Use namespace to avoid Jinja loop-scoping bug with set inside for. #}
            {% set found = namespace(outer=none, inner=none) %}
            {% for p in (structured_parse.pieces or []) %}
                {% if p.id == op.outer_piece_id %}{% set found.outer = p %}{% endif %}
                {% if p.id == op.inner_piece_id %}{% set found.inner = p %}{% endif %}
            {% endfor %}
            {% if found.outer and found.inner %}
            <span class="font-mono text-sky-700">{{ found.outer.letters }}</span>
            <span class="text-gray-400">around</span>
            <span class="font-mono text-pink-700">{{ found.inner.letters }}</span>
            <span class="text-gray-400">=</span>
            <span class="font-mono text-emerald-700">{{ op.result or structured_parse.answer }}</span>
            {% endif %}
        {% endif %}
    </div>
    {% endfor %}

</div>

<form hx-post="/admin/structured-parse/{{ clue_id }}/delete"
      hx-target="#manual-evidence-list-{{ clue_id }}"
      hx-swap="outerHTML"
      class="mt-1">
    <button type="submit"
            class="text-xs text-red-400 hover:text-red-600 cursor-pointer"
            title="Delete this structured parse">
        Delete structured parse
    </button>
</form>

{% elif nodes %}

{# === LEGACY GRAPH DISPLAY === #}
<p class="text-xs font-semibold text-gray-500 mt-1 mb-1">Saved clue evidence</p>
<ul class="space-y-1 mt-1">
{% for node in nodes %}
<li class="flex flex-wrap items-center gap-2 rounded border bg-white px-2 py-1.5 text-xs
    {% if node.node_type == 'source' %}border-sky-200
    {% elif node.node_type == 'definition' %}border-green-200
    {% elif node.node_type == 'operator' %}border-amber-200
    {% elif node.node_type == 'transform' %}border-violet-200
    {% else %}border-slate-200{% endif %}">
    <span class="font-mono text-gray-400 shrink-0" title="Reference number for connecting evidence">#{{ node.id }}</span>

    {% if node.node_type == 'source' %}
    <span class="rounded bg-sky-100 px-1.5 py-0.5 font-semibold text-sky-700 shrink-0">clue letters</span>
    <span class="font-medium text-gray-900">{{ node.word_text }}</span>
    {% if node.role %}
    <span class="text-gray-500">
        {% if node.role == 'synonym' %}means
        {% elif node.role == 'abbreviation' %}abbreviates to
        {% elif node.role == 'literal' %}gives
        {% elif node.role == 'pronoun' %}stands for
        {% elif node.role == 'foreign' %}translates to
        {% else %}gives
        {% endif %}
    </span>
    {% else %}
    <span class="text-gray-500">gives</span>
    {% endif %}
    <span class="font-mono text-emerald-700">{{ node.raw_letters or '?' }}</span>

    {% elif node.node_type == 'operator' %}
    <span class="rounded bg-amber-100 px-1.5 py-0.5 font-semibold text-amber-700 shrink-0">operation</span>
    <span class="font-medium text-gray-900">{{ node.word_text }}</span>
    <span class="text-gray-500">
        {% if node.role == 'reversal' %}reverses letters
        {% elif node.role == 'deletion' %}deletes letters
        {% elif node.role == 'anagram' %}rearranges letters
        {% elif node.role == 'container' %}puts one part inside another
        {% else %}{{ node.role or 'operation' }}
        {% endif %}
    </span>

    {% elif node.node_type == 'transform' %}
    <span class="rounded bg-violet-100 px-1.5 py-0.5 font-semibold text-violet-700 shrink-0">operation result</span>
    <span class="font-medium text-gray-900">{{ node.word_text or 'result' }}</span>
    <span class="text-gray-500">makes</span>
    <span class="font-mono text-emerald-700">{{ node.raw_letters or '?' }}</span>

    {% elif node.node_type == 'definition' %}
    <span class="rounded bg-green-100 px-1.5 py-0.5 font-semibold text-green-700 shrink-0">definition</span>
    <span class="font-medium text-gray-900">{{ node.word_text }}</span>

    {% elif node.node_type == 'structural' %}
    <span class="rounded bg-slate-100 px-1.5 py-0.5 font-semibold text-slate-600 shrink-0">link/filler</span>
    <span class="font-medium text-gray-900">{{ node.word_text }}</span>

    {% else %}
    <span class="rounded bg-slate-100 px-1.5 py-0.5 font-semibold text-slate-600 shrink-0">saved</span>
    <span class="font-medium text-gray-900">{{ node.word_text }}</span>
    {% endif %}

    {% if node.answer_positions is not none %}
    <span class="rounded bg-sky-50 px-1.5 py-0.5 font-mono text-sky-700">answer {{ node.answer_positions }}</span>
    {% endif %}
    {% if node.group_id is not none %}
    <span class="rounded bg-violet-50 px-1.5 py-0.5 text-violet-700">colour {{ node.group_id }}</span>
    {% endif %}
    {% if node.word_indices %}
    <span class="rounded bg-gray-50 px-1.5 py-0.5 text-gray-500">words {{ node.word_indices }}</span>
    {% endif %}
    {% if node.payload_json %}
    <details class="text-gray-500">
        <summary class="cursor-pointer hover:text-gray-700">extra details</summary>
        <span class="font-mono text-[11px] text-orange-700">{{ node.payload_json }}</span>
    </details>
    {% endif %}
    <form hx-post="/admin/manual-evidence/node/{{ node.id }}/delete"
          hx-target="#manual-evidence-list-{{ clue_id }}"
          hx-swap="outerHTML"
          class="ml-auto">
        <button type="submit"
                class="text-red-400 hover:text-red-600 font-bold cursor-pointer"
                title="Delete this saved evidence">x</button>
    </form>
</li>
{% endfor %}
</ul>

{% if edges %}
<p class="text-xs font-semibold text-gray-500 mt-3 mb-1">Saved connections</p>
<ul class="space-y-1 mt-1">
{% for edge in edges %}
{% set from_label = namespace(text='#' ~ edge.from_node_id) %}
{% set to_label = namespace(text='#' ~ edge.to_node_id) %}
{% for node in nodes %}
    {% if node.id == edge.from_node_id %}
        {% if node.raw_letters %}
            {% set from_label.text = node.word_text ~ ' -> ' ~ node.raw_letters %}
        {% else %}
            {% set from_label.text = node.word_text %}
        {% endif %}
    {% endif %}
    {% if node.id == edge.to_node_id %}
        {% if node.raw_letters %}
            {% set to_label.text = node.word_text ~ ' -> ' ~ node.raw_letters %}
        {% else %}
            {% set to_label.text = node.word_text %}
        {% endif %}
    {% endif %}
{% endfor %}
<li class="flex flex-wrap items-center gap-2 rounded border border-gray-100 bg-white px-2 py-1 text-xs">
    <span class="font-mono text-gray-400" title="Connection reference">#{{ edge.id }}</span>
    {% if edge.edge_type == 'input_to' %}
    <span class="font-medium text-gray-900">{{ from_label.text }}</span>
    <span class="text-gray-500">feeds</span>
    <span class="font-medium text-gray-900">{{ to_label.text }}</span>
    {% elif edge.edge_type == 'output_of' %}
    <span class="font-medium text-gray-900">{{ from_label.text }}</span>
    <span class="text-gray-500">is the result of</span>
    <span class="font-medium text-gray-900">{{ to_label.text }}</span>
    {% else %}
    <span class="font-medium text-gray-900">{{ from_label.text }}</span>
    <span class="text-gray-500">connects to</span>
    <span class="font-medium text-gray-900">{{ to_label.text }}</span>
    {% endif %}
    <form hx-post="/admin/manual-evidence/edge/{{ edge.id }}/delete"
          hx-target="#manual-evidence-list-{{ clue_id }}"
          hx-swap="outerHTML"
          class="ml-auto">
        <button type="submit"
                class="text-red-400 hover:text-red-600 font-bold cursor-pointer"
                title="Delete this connection">x</button>
    </form>
</li>
{% endfor %}
</ul>
{% endif %}

{% else %}
<p class="text-xs text-gray-400 mt-1">No manual evidence recorded yet.</p>
{% endif %}

</div>
```

---

## 6. Boundary invariants (must hold after implementation)

These invariants must be satisfied everywhere in the new code:

1. `answer_boxes` in `parse_dict` are always 1-based. No stored JSON contains
   0-based answer_boxes.
2. `answer_positions` in `manual_evidence_nodes` (legacy) are always 0-based.
   These two systems never mix.
3. In `merge_structured_parse_into_display`, the conversion `idx = box - 1`
   appears exactly once per box, at the point of indexing `cleaned[idx]`.
4. `manual_structured_parses` is written only from the structured parse routes.
   The graph node routes never write to it.
5. `merge_structured_parse_into_display` never reads from `manual_evidence_nodes`
   or `manual_evidence_edges`.
6. `merge_manual_into_display` (legacy) is not modified.

---

## 7. What is NOT in this slice

Do not implement any of the following. They are deferred:

- Reversal, anagram, deletion, homophone, charade editor forms
- Multi-step operations (transform_pieces)
- LLM JSON import endpoint
- Click-to-fill word positions
- Promotion to wfw_proof_attempts
- Any public-facing change

---

## 8. SWALLOW acceptance test

Clue: `Bird, female, going over fence (7)`
Answer: `SWALLOW`

Cleaned answer: `SWALLOW` (7 letters)

```
S W A L L O W
1 2 3 4 5 6 7
```

### 8.1 Form entry

Definition: `Bird`
Piece 1 (outer): clue words `female`, relationship `synonym`, letters `SOW`,
  answer boxes `1,6,7`, colour `blue`
Piece 2 (inner): clue words `fence`, relationship `synonym`, letters `WALL`,
  answer boxes `2,3,4,5`, colour `pink`
Container indicator: clue words `going over`, outer piece = Piece 1

### 8.2 Expected validation

`_validate_structured_parse` must return `[]` (no errors) for this input.

Check:
- SOW at [1,6,7]: `S`=`SWALLOW[0]` ✓, `O`=`SWALLOW[5]` ✓, `W`=`SWALLOW[6]` ✓
- WALL at [2,3,4,5]: `W`=`SWALLOW[1]` ✓, `A`=`SWALLOW[2]` ✓, `L`=`SWALLOW[3]` ✓,
  `L`=`SWALLOW[4]` ✓
- Container: `WALL` in `SWALLOW` at index 1; removing it leaves `SOW` ✓;
  result `SWALLOW` == cleaned answer ✓

### 8.3 Expected display after save

Clue blocks:
- `Bird` — DEF_BLOCK, role `definition`
- `female` — SOURCE_BLOCK, role `piece_0` (blue)
- `going over` — OP_BLOCK, role `container_indicator`, status `manual`
- `fence` — SOURCE_BLOCK, role `piece_1` (pink)

Answer links (0-based index → role):
- index 0 (S): `piece_0` (blue), source_text `female`
- index 1 (W): `piece_1` (pink), source_text `fence`
- index 2 (A): `piece_1` (pink), source_text `fence`
- index 3 (L): `piece_1` (pink), source_text `fence`
- index 4 (L): `piece_1` (pink), source_text `fence`
- index 5 (O): `piece_0` (blue), source_text `female`
- index 6 (W): `piece_0` (blue), source_text `female`

### 8.4 Rerun invariant

After a pipeline rerun, `manual_structured_parses` still has the SWALLOW row.
The graph tables may have changed but the structured parse path is unaffected.

---

## 9. Validation test cases

### Test: wrong letter assignment

Piece 1 `female` → `SOW`, boxes `[1,5,7]`.

`SWALLOW[4]` = `L`, not `O`.

`_validate_structured_parse` must return an error containing:
`"piece piece1: letter 'O' at box 5 does not match answer letter 'L'"`

### Test: duplicate box

Two pieces both claim box 2.

`_validate_structured_parse` must return an error containing:
`"box 2 claimed by both"`

### Test: container inner not in result

Outer `SOW`, inner `WALL`, result `SWOLLAW`.

`WALL` is not a substring of `SWOLLAW`.

`_validate_structured_parse` must return an error containing:
`"inner 'WALL' not found as substring in result 'SWOLLAW'"`

---

## 10. Verification checklist (run before shipping)

Run each check. Do not mark the slice complete until all pass.

```
CHECK 1 — Python syntax
  python -m py_compile signature_solver/manual_evidence_store.py
  python -m py_compile web/routes/admin.py
  python -m py_compile web/routes/clue.py
  Expected: no output (exit 0 for each)

CHECK 2 — Table round-trip
  python - <<'EOF'
  import sqlite3, json, tempfile, os
  from pathlib import Path
  import sys
  sys.path.insert(0, '.')
  # Patch CLUES_DB to a temp file
  import signature_solver.manual_evidence_store as s
  tmp = tempfile.mktemp(suffix='.db')
  s.CLUES_DB = Path(tmp)
  conn = sqlite3.connect(tmp)
  s.ensure_tables(conn)
  conn.commit()
  # Check table exists
  tables = {r[0] for r in conn.execute(
      "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
  assert 'manual_structured_parses' in tables, tables
  # Write and read back
  parse = {"version":1,"clue_id":99,"answer":"SWALLOW","pieces":[]}
  s.write_structured_parse(99, parse, conn=conn)
  conn.commit()
  result = s.get_structured_parse(99, conn=conn)
  assert result["answer"] == "SWALLOW", result
  conn.close()
  os.unlink(tmp)
  print("CHECK 2 PASS")
  EOF

CHECK 3 — SWALLOW validation returns no errors
  python - <<'EOF'
  import sys; sys.path.insert(0, '.')
  from signature_solver.manual_evidence_store import _validate_structured_parse
  parse = {
    "version": 1,
    "definition": {"id":"def1","clue_text":"Bird","clue_word_positions":[0],"answer":"SWALLOW"},
    "pieces": [
      {"id":"piece1","clue_text":"female","clue_word_positions":[1],
       "relationship":"synonym","letters":"SOW","answer_boxes":[1,6,7],
       "mapping":"positional","colour":"blue"},
      {"id":"piece2","clue_text":"fence","clue_word_positions":[4],
       "relationship":"synonym","letters":"WALL","answer_boxes":[2,3,4,5],
       "mapping":"positional","colour":"pink"},
    ],
    "operations": [
      {"id":"op1","clue_text":"going over","clue_word_positions":[2,3],
       "type":"container","outer_piece_id":"piece1","inner_piece_id":"piece2",
       "result":"SWALLOW"}
    ],
    "filler": [],
  }
  errors = _validate_structured_parse(parse, "SWALLOW")
  assert errors == [], "Expected no errors, got: %s" % errors
  print("CHECK 3 PASS")
  EOF

CHECK 4 — SWALLOW merge produces correct answer_links
  python - <<'EOF'
  import sys; sys.path.insert(0, '.')
  from signature_solver.manual_evidence_store import merge_structured_parse_into_display
  parse = {
    "version": 1,
    "definition": {"id":"def1","clue_text":"Bird","clue_word_positions":[0],"answer":"SWALLOW"},
    "pieces": [
      {"id":"piece1","clue_text":"female","clue_word_positions":[1],
       "relationship":"synonym","letters":"SOW","answer_boxes":[1,6,7],
       "mapping":"positional","colour":"blue"},
      {"id":"piece2","clue_text":"fence","clue_word_positions":[4],
       "relationship":"synonym","letters":"WALL","answer_boxes":[2,3,4,5],
       "mapping":"positional","colour":"pink"},
    ],
    "operations": [
      {"id":"op1","clue_text":"going over","clue_word_positions":[2,3],
       "type":"container","outer_piece_id":"piece1","inner_piece_id":"piece2",
       "result":"SWALLOW"}
    ],
    "filler": [],
  }
  display = {"blocks": [], "answer_links": []}
  merge_structured_parse_into_display(display, parse, "SWALLOW")
  links = display["answer_links"]
  assert len(links) == 7, "Expected 7 links, got %d" % len(links)
  # S at index 0 should be blue (piece_0)
  assert links[0]["source_role"] == "piece_0", links[0]
  # W at index 1 should be pink (piece_1)
  assert links[1]["source_role"] == "piece_1", links[1]
  # L at index 4 should be pink (piece_1)
  assert links[4]["source_role"] == "piece_1", links[4]
  # O at index 5 should be blue (piece_0)
  assert links[5]["source_role"] == "piece_0", links[5]
  # W at index 6 should be blue (piece_0)
  assert links[6]["source_role"] == "piece_0", links[6]
  print("CHECK 4 PASS")
  EOF

CHECK 5 — Wrong letter validation returns error
  python - <<'EOF'
  import sys; sys.path.insert(0, '.')
  from signature_solver.manual_evidence_store import _validate_structured_parse
  parse = {
    "version": 1,
    "definition": {"id":"def1","clue_text":"Bird","clue_word_positions":[0],"answer":"SWALLOW"},
    "pieces": [
      {"id":"piece1","clue_text":"female","clue_word_positions":[1],
       "relationship":"synonym","letters":"SOW","answer_boxes":[1,5,7],
       "mapping":"positional","colour":"blue"},
    ],
    "operations": [],
    "filler": [],
  }
  errors = _validate_structured_parse(parse, "SWALLOW")
  assert any("box 5" in e for e in errors), "Expected box 5 error, got: %s" % errors
  print("CHECK 5 PASS")
  EOF

CHECK 6 — Rerun audit
  Confirm that manual_structured_parses table is NOT deleted or cleared by any
  rerun or reverify code path. Search for any script that truncates or drops
  tables in data/clues_master.db and verify manual_structured_parses is not
  affected. This is a manual code-read check, not a script.

CHECK 7 — Jinja2 template syntax
  python - <<'EOF'
  from jinja2 import Environment, FileSystemLoader
  env = Environment(loader=FileSystemLoader('web/templates'))
  env.get_template('clue.html')
  env.get_template('partials/manual_evidence_nodes.html')
  print("CHECK 7 PASS")
  EOF
```

---

## 11. File touch summary

| File | Change |
|------|--------|
| `signature_solver/manual_evidence_store.py` | Append new code (DDL, store functions, validator, merge) |
| `web/routes/clue.py` | Replace admin merge block (lines ~452–489) |
| `web/routes/admin.py` | Add `_render_evidence_partial`; add 2 new routes; update 4 existing routes |
| `web/templates/clue.html` | Replace admin editor inner div content |
| `web/templates/partials/manual_evidence_nodes.html` | Replace entire file |

No other files are touched.
