# Phase 2 Manual Evidence Graph — First Slice Codex Instruction

Date: 2026-05-26
Status: ready for implementation

---

## Task

Build the first slice of the persistent manual evidence graph. This lets an
admin record direct source placements, definition spans, and structural words
for any clue. The evidence is stored durably in clues_master.db, survives
rerun, and is merged into the clue display at page load time.

---

## Scope of this slice

### What this slice does

- Creates two new tables in clues_master.db: manual_evidence_nodes and
  manual_evidence_edges (edges table created now, unused until a later slice)
- New module: signature_solver/manual_evidence_store.py with storage functions
  and a display merge function
- Two new admin routes in web/routes/admin.py: POST to create a node, POST to
  delete a node
- Display-time merge in web/routes/clue.py: at page load for admin requests,
  manual nodes are loaded and merged into the display struct in memory

### Publication behaviour — read this carefully

Slice 1 is admin-visible display merge only.

The merge is gated on g.is_admin. Public visitors see the unmodified automatic
display. The wfw_proof_attempts table is not touched. The clue's proof status
in the database is not changed. Nothing Slice 1 does will alter what an
anonymous visitor sees on a clue page.

This remains true unless and until later proof-status work explicitly promotes
manual evidence into wfw_proof_attempts and makes the resulting proof the
public-facing authority for that clue.

### What this slice defers

- Operator nodes (reversal, deletion, anagram, container, homophone, hidden)
- Transform nodes
- Assembly nodes
- Edges (manual_evidence_edges table is created but no edges are written or read)

---

## Files that change

Five files change:

1. signature_solver/manual_evidence_store.py — new file
2. web/routes/admin.py — two new routes inserted after save_wfw_correction
3. web/routes/clue.py — load manual nodes and call merge (admin only)
4. web/templates/clue.html — new admin section
5. web/templates/partials/manual_evidence_nodes.html — new partial template

---

## Change 1: signature_solver/manual_evidence_store.py (new file)

Create this file with the full content below.

### Design notes for merge_manual_into_display

Answer cleaning: answer_positions are validated against the alpha-only version
of the answer string. Hyphens, spaces, and punctuation in the stored answer are
stripped before checking. _clean_answer(answer) does this stripping.

Letter validation: for each source node with raw_letters and answer_positions,
_validate_source_letters checks that len(raw_letters) == len(answer_positions)
and that raw_letters[i] == cleaned_answer[answer_positions[i]] for every i.
If validation fails, the SOURCE_BLOCK is marked evidence_status="failed" and
role="source_review" (rose/red). No answer links are emitted for that node.
Validation is skipped (treated as passing) when raw_letters is absent,
answer_positions is absent or empty, or the cleaned answer is empty.

Colour role from group_id: when a source node carries group_id G, its
SOURCE_BLOCK role is "piece_{G}" and its answer_link source_role is "piece_{G}".
This makes the clue block and the answer tiles share the same colour. The
mapping in atomic_parse.html is: piece_0=blue, piece_1=pink, piece_2=yellow,
piece_3=orange, piece_4=purple. group_id is validated to the range 0-4 in the
admin route (see Change 2). Nodes without group_id use their semantic role.

Auto SOURCE_BLOCK suppression: in addition to suppressing REVIEW_BLOCKs,
merge_manual_into_display also suppresses automatic SOURCE_BLOCKs whose span
overlaps the covered_word_indices of any manual node. This prevents the
automatic solver's evidence from appearing alongside or overriding manual
evidence for the same clue words.

Complete answer row: merge_manual_into_display builds the final answer_links
as a single authoritative position map (a dict keyed by answer_index) to
guarantee exactly one tile per answer letter with no duplicates possible.

Step 1 builds manual_by_pos from valid source nodes. If two manual nodes
claim the same position, both entries are discarded for that position and it
is recorded as conflicted; conflicted positions receive a plain entry
(source_role=None, white tile). Out-of-range positions (p < 0 or p >= len
cleaned) are silently ignored. Step 2 builds auto_by_pos from the existing
wfw_display["answer_links"], including only positions not already in
manual_by_pos, taking first occurrence at each position. Step 3 iterates
over enumerate(cleaned) and emits exactly one entry per position:
manual_by_pos takes priority, then auto_by_pos, then a plain entry. The
result is always len(cleaned) entries, in order 0..len(cleaned)-1, with no
duplicate positions possible. The covered_answer_positions variable from the
earlier list-based approach is not used and is removed entirely.

### Full file content

```python
"""Persistence and display merge for manual evidence nodes.

Manual evidence is stored separately from automatic proof attempts.
It survives rerun and is merged into the final clue display at load time.

This is display-merge only (Slice 1). Nothing here writes to
wfw_proof_attempts or changes the clue's proof status in the database.
The merge is visible to admin users only. Public display is unchanged.

Only source, definition, and structural nodes are used in the first slice.
Operator, transform, and assembly nodes are deferred.
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
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_node(clue_id, node_type, word_indices, word_text,
               role=None, raw_letters=None, answer_positions=None,
               group_id=None, conn=None):
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
        cursor = conn.execute(
            """INSERT INTO manual_evidence_nodes
               (clue_id, node_type, word_indices, word_text, role,
                raw_letters, answer_positions, group_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                clue_id,
                node_type,
                json.dumps(list(word_indices)),
                word_text,
                role or None,
                letters,
                positions_json,
                group_id,
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


def get_nodes_for_clue(clue_id, conn=None):
    """Return all manual evidence nodes for a clue as a list of dicts.

    word_indices is decoded from JSON to a Python list of ints.
    answer_positions is decoded from JSON to a Python list of ints,
    or None if not set in the DB.

    Returns [] if the table does not exist yet or on any read error.
    """
    own = conn is None
    if own:
        conn = sqlite3.connect(
            f"file:{CLUES_DB}?mode=ro", uri=True, timeout=30
        )
        conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, clue_id, node_type, word_indices, word_text,
                      role, raw_letters, answer_positions, group_id,
                      source, created_at
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
            result.append(d)
        return result
    except Exception:
        return []
    finally:
        if own:
            conn.close()


# ---------------------------------------------------------------------------
# Display merge helpers
# ---------------------------------------------------------------------------

def _clean_answer(answer):
    """Return only alpha characters from answer, uppercased.

    answer_positions in manual nodes refer to positions in this cleaned
    string, not in the raw stored answer.
    """
    return re.sub(r"[^A-Za-z]", "", answer or "").upper()


def _colour_role(node):
    """Return the display role string for a source node.

    If group_id is set (0-4), returns "piece_{group_id}" which maps to a
    named colour in atomic_parse.html. Otherwise returns the semantic role
    or 'synonym' as a fallback (white tile, no special colour).
    """
    gid = node.get("group_id")
    if gid is not None:
        return "piece_%d" % gid
    return node.get("role") or "synonym"


def _validate_source_letters(node, answer):
    """Return True if raw_letters matches the cleaned answer at answer_positions.

    Skips validation (returns True) when raw_letters is absent,
    answer_positions is absent or empty, or the cleaned answer is empty.
    """
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


# ---------------------------------------------------------------------------
# Display merge
# ---------------------------------------------------------------------------

def merge_manual_into_display(wfw_display, manual_nodes, answer):
    """Merge manual evidence nodes into an existing wfw_display dict.

    Mutates wfw_display["blocks"] and wfw_display["answer_links"] in place.
    Has no Flask dependencies.

    This is display-merge only. It does not write to wfw_proof_attempts or
    change any database state. It is only called for admin page requests.

    Source nodes:
      - Validated against cleaned answer at claimed positions.
      - Pass: SOURCE_BLOCK with colour role from group_id; answer links at
        each claimed position.
      - Fail: SOURCE_BLOCK with role="source_review" (rose) and
        evidence_status="failed"; no answer links for this node.

    Definition nodes:
      - DEF_BLOCK added.
      - All existing auto DEF_BLOCKs removed.

    Structural nodes:
      - No block added; word_indices counted as covered to suppress
        REVIEW_BLOCKs and auto SOURCE_BLOCKs at those positions.

    All nodes:
      - REVIEW_BLOCKs whose span overlaps covered_word_indices are removed.
      - Auto SOURCE_BLOCKs whose span overlaps covered_word_indices are
        removed.

    Complete answer row:
      - answer_links is built as an authoritative position map keyed by
        answer_index, guaranteeing exactly one tile per cleaned-answer
        position. Two manual nodes claiming the same position produce a
        plain entry at that position (conflict). Auto links fill unclaimed
        positions. Remaining positions receive plain entries (white tiles).
        The result is always len(cleaned) entries with no duplicate positions.
    """
    cleaned = _clean_answer(answer)

    # Collect word indices covered by all manual nodes.
    covered_word_indices = set()
    for node in manual_nodes:
        for idx in (node.get("word_indices") or []):
            covered_word_indices.add(idx)

    has_manual_def = any(
        n.get("node_type") == "definition" for n in manual_nodes
    )

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

    # 2. Manual SOURCE_BLOCKs.
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

    # 3. Keep existing auto blocks with suppression rules.
    for block in (wfw_display.get("blocks") or []):
        kind = block.get("kind")
        span = block.get("span") or []
        span_indices = (
            set(range(span[0], span[1]))
            if isinstance(span, list) and len(span) == 2
            else set()
        )

        # Suppress all auto DEF_BLOCKs when any manual definition exists.
        if kind == "DEF_BLOCK" and has_manual_def:
            continue

        # Suppress auto SOURCE_BLOCKs whose span overlaps covered words.
        if kind == "SOURCE_BLOCK" and span_indices & covered_word_indices:
            continue

        # Suppress REVIEW_BLOCKs whose span overlaps covered words.
        if kind == "REVIEW_BLOCK" and span_indices & covered_word_indices:
            continue

        new_blocks.append(block)

    wfw_display["blocks"] = new_blocks

    # Build answer_links as an authoritative position map.
    # Step 1: manual_by_pos from valid source nodes.
    # Two nodes claiming the same position → conflict → plain entry at that position.
    # Out-of-range positions are silently ignored.
    manual_by_pos = {}
    conflicted = set()
    for node in manual_nodes:
        if node.get("node_type") != "source":
            continue
        if not _validate_source_letters(node, answer):
            continue
        wi = node.get("word_indices") or []
        span = [min(wi), max(wi) + 1] if wi else [0, 1]
        role = _colour_role(node)
        for p in (node.get("answer_positions") or []):
            if p < 0 or p >= len(cleaned):
                continue
            if p in manual_by_pos:
                conflicted.add(p)
            else:
                manual_by_pos[p] = {
                    "answer_index": p,
                    "letter": cleaned[p],
                    "source_block": "manual_src_%d" % node["id"],
                    "source_span": span,
                    "source_text": node["word_text"],
                    "source_role": role,
                    "source_value": node.get("raw_letters") or "",
                    "source_input_value": "",
                    "source_value_index": None,
                }
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

    # Step 2: auto_by_pos from existing answer_links.
    # Include only in-range positions not already in manual_by_pos,
    # first occurrence wins.
    auto_by_pos = {}
    for link in (wfw_display.get("answer_links") or []):
        p = link.get("answer_index")
        if p is None or p < 0 or p >= len(cleaned):
            continue
        if p not in manual_by_pos and p not in auto_by_pos:
            auto_by_pos[p] = link

    # Step 3: emit exactly one entry per cleaned-answer position.
    # Priority: manual_by_pos (includes conflicted-plain) > auto_by_pos > plain.
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

Insert the two route functions below immediately after the save_wfw_correction
route function (which ends just before the edit_form route at line 626).
Do not change any other part of admin.py.

group_id is validated server-side to the range 0-4. A value outside that
range returns HTTP 400.

```python
@bp.route("/manual-evidence/<int:clue_id>", methods=["POST"])
def create_manual_evidence_node(clue_id):
    """Create a manual evidence node for a clue."""
    _require_admin()
    db = get_admin_db()
    clue = db.execute(
        "SELECT id FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    if clue is None:
        abort(404)

    word_text = (request.form.get("word_text") or "").strip()
    word_indices_raw = (request.form.get("word_indices") or "").strip()
    node_type = (request.form.get("node_type") or "").strip()
    role = (request.form.get("role") or "").strip() or None
    raw_letters = (
        (request.form.get("raw_letters") or "").strip().upper() or None
    )
    answer_positions_raw = (
        request.form.get("answer_positions") or ""
    ).strip()
    group_id_raw = (request.form.get("group_id") or "").strip()

    if node_type not in ("source", "definition", "structural"):
        abort(400)
    if not word_text:
        abort(400)

    try:
        word_indices = [
            int(x.strip())
            for x in word_indices_raw.split(",")
            if x.strip()
        ]
    except (ValueError, TypeError):
        word_indices = []
    if not word_indices:
        abort(400)

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

    group_id = None
    if group_id_raw:
        try:
            group_id = int(group_id_raw)
        except (ValueError, TypeError):
            abort(400)
        if group_id < 0 or group_id > 4:
            abort(400)

    from signature_solver.manual_evidence_store import (
        write_node, get_nodes_for_clue,
    )
    write_node(
        clue_id=clue_id,
        node_type=node_type,
        word_indices=word_indices,
        word_text=word_text,
        role=role,
        raw_letters=raw_letters,
        answer_positions=answer_positions,
        group_id=group_id,
        conn=db,
    )
    db.commit()
    nodes = get_nodes_for_clue(clue_id, conn=db)
    return render_template(
        "partials/manual_evidence_nodes.html",
        clue_id=clue_id,
        nodes=nodes,
    )


@bp.route("/manual-evidence/node/<int:node_id>/delete", methods=["POST"])
def delete_manual_evidence_node(node_id):
    """Delete a manual evidence node."""
    _require_admin()
    db = get_admin_db()
    row = db.execute(
        "SELECT clue_id FROM manual_evidence_nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    if row is None:
        abort(404)
    clue_id = row["clue_id"]

    from signature_solver.manual_evidence_store import (
        delete_node, get_nodes_for_clue,
    )
    delete_node(node_id, conn=db)
    db.commit()
    nodes = get_nodes_for_clue(clue_id, conn=db)
    return render_template(
        "partials/manual_evidence_nodes.html",
        clue_id=clue_id,
        nodes=nodes,
    )
```

---

## Change 3: web/routes/clue.py

Locate this exact line in clue_page:

    clue_dict["stage_two_casefile"] = None

It is the first line after the blank line following the wfw_display try/except
block (current line 448). Insert the following block immediately before it.

g is already imported in clue.py (from flask import ... g ...) so no new
import is needed.

```python
    # Manual evidence: load from DB and merge into wfw_display for admins.
    # This is display-merge only (Slice 1).
    # The merge is gated on is_admin: public visitors see the unmodified
    # automatic display. wfw_proof_attempts is not touched.
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
            )
    clue_dict["manual_evidence_nodes"] = _manual_nodes
```

clue_dict["manual_evidence_nodes"] is set unconditionally (empty list for
non-admins) so the template can always reference clue.manual_evidence_nodes
without a KeyError.

No other changes to clue.py.

---

## Change 4: web/templates/clue.html

Locate the line:

    {% endif %}

that closes the `{% if clue.wfw_role_rows %}` block (currently line 374).
It is immediately followed by a blank line and then:

    <details class="mt-3">

which opens "Admin: add WFW data". Insert the block below between that
`{% endif %}` and the `<details class="mt-3">`:

```html
    {# Admin: manual evidence graph — source, definition, structural nodes #}
    <details class="mt-3">
        <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-700">Admin: manual evidence</summary>
        <div class="mt-2 rounded border border-violet-200 bg-violet-50 px-4 py-3">
            <p class="mb-2 text-xs text-violet-700">
                Word indices: 0-based word positions in the clue (0 = first word).
                Answer positions: 0-based character positions in the alpha-only answer.
                Group id drives tile colour: 0=blue, 1=pink, 2=yellow, 3=orange, 4=purple.
                Leave group id blank for no colour coding.
            </p>
            <form hx-post="/admin/manual-evidence/{{ clue.id }}"
                  hx-target="#manual-evidence-list-{{ clue.id }}"
                  hx-swap="outerHTML"
                  hx-on::after-request="this.reset()"
                  class="grid grid-cols-1 gap-2 text-sm mb-3">
                <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Clue span text
                        <input name="word_text" placeholder="That man" required
                               class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                    </label>
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Word indices (0-based, comma sep)
                        <input name="word_indices" placeholder="0,1" required
                               class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                    </label>
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Node type
                        <select name="node_type"
                                class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                            <option value="source">source</option>
                            <option value="definition">definition</option>
                            <option value="structural">structural</option>
                        </select>
                    </label>
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Role (optional)
                        <input name="role" placeholder="synonym"
                               class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                    </label>
                </div>
                <div class="grid grid-cols-2 md:grid-cols-3 gap-2">
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Letters produced (source only)
                        <input name="raw_letters" placeholder="HE"
                               class="rounded border border-slate-300 px-2 py-1 bg-white uppercase font-mono text-xs">
                    </label>
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Answer positions (comma sep, optional)
                        <input name="answer_positions" placeholder="0,1"
                               class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                    </label>
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Group id (0-4, optional)
                        <input name="group_id" type="number" min="0" max="4" placeholder=""
                               class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                    </label>
                </div>
                <button type="submit"
                        class="self-start text-xs px-3 py-1 rounded border border-violet-400 text-violet-700 hover:bg-violet-100 cursor-pointer">
                    Add evidence node
                </button>
            </form>
            {% with clue_id=clue.id, nodes=clue.manual_evidence_nodes %}
            {% include "partials/manual_evidence_nodes.html" %}
            {% endwith %}
        </div>
    </details>
```

---

## Change 5: web/templates/partials/manual_evidence_nodes.html (new file)

Variables expected: clue_id (int), nodes (list of dicts from get_nodes_for_clue).

```html
{# Manual evidence node list.
   Included in clue.html and returned by create/delete admin routes.
   Variables: clue_id (int), nodes (list of dicts). #}
<div id="manual-evidence-list-{{ clue_id }}">
{% if nodes %}
<ul class="space-y-1 mt-2">
{% for node in nodes %}
<li class="flex flex-wrap items-center gap-2 rounded border bg-white px-2 py-1.5 text-xs
    {% if node.node_type == 'source' %}border-sky-200
    {% elif node.node_type == 'definition' %}border-green-200
    {% else %}border-slate-200{% endif %}">
    <span class="rounded px-1.5 py-0.5 font-semibold shrink-0
        {% if node.node_type == 'source' %}bg-sky-100 text-sky-700
        {% elif node.node_type == 'definition' %}bg-green-100 text-green-700
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
    <span class="text-gray-400">idx {{ node.word_indices }}</span>
    <form hx-post="/admin/manual-evidence/node/{{ node.id }}/delete"
          hx-target="#manual-evidence-list-{{ clue_id }}"
          hx-swap="outerHTML"
          class="ml-auto">
        <button type="submit"
                class="text-red-400 hover:text-red-600 font-bold cursor-pointer"
                title="Delete this node">&#10005;</button>
    </form>
</li>
{% endfor %}
</ul>
{% else %}
<p class="text-xs text-gray-400 mt-1">No manual evidence recorded.</p>
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

### Check 2 — store round-trip using a temporary database

Uses a temp file. No data reaches clues_master.db.

    .venv\Scripts\python.exe -c "
    import sys, tempfile, os, sqlite3
    sys.path.insert(0, '.')
    from signature_solver.manual_evidence_store import (
        write_node, get_nodes_for_clue, delete_node
    )
    tmp = tempfile.mktemp(suffix='.db')
    try:
        conn = sqlite3.connect(tmp)
        conn.row_factory = sqlite3.Row

        node_id = write_node(
            clue_id=1,
            node_type='source',
            word_indices=[0],
            word_text='test_span',
            role='synonym',
            raw_letters='TEST',
            answer_positions=[0, 1, 2, 3],
            group_id=0,
            conn=conn,
        )
        conn.commit()
        print('Created node id:', node_id)

        nodes = get_nodes_for_clue(1, conn=conn)
        found = next((n for n in nodes if n['id'] == node_id), None)
        assert found is not None, 'Node not found after write'
        assert found['raw_letters'] == 'TEST', 'Letters mismatch'
        assert found['word_indices'] == [0], 'word_indices mismatch'
        assert found['answer_positions'] == [0, 1, 2, 3], 'positions mismatch'
        assert found['group_id'] == 0, 'group_id mismatch'
        print('PASS: node stored and retrieved correctly')

        delete_node(node_id, conn=conn)
        conn.commit()
        nodes_after = get_nodes_for_clue(1, conn=conn)
        assert not any(n['id'] == node_id for n in nodes_after), 'Node not deleted'
        print('PASS: node deleted correctly')
    finally:
        conn.close()
        try:
            os.unlink(tmp)
        except Exception:
            pass
    "

Expected output:

    Created node id: <some integer>
    PASS: node stored and retrieved correctly
    PASS: node deleted correctly

### Check 3 — merge function: four scenarios without Flask or live DB

    .venv\Scripts\python.exe -c "
    import sys; sys.path.insert(0, '.')
    from signature_solver.manual_evidence_store import merge_manual_into_display

    # --- Scenario A: suppression and DEF_BLOCK override ---
    # Source text ('that') and manual definition text ('manual definition')
    # are deliberately different from the auto DEF_BLOCK text ('auto def')
    # so each can be located unambiguously by text.
    wfw = {
        'blocks': [
            {'kind': 'REVIEW_BLOCK', 'span': [0, 1], 'text': 'that', 'role': 'unaccounted'},
            {'kind': 'SOURCE_BLOCK', 'span': [0, 1], 'text': 'that', 'role': 'piece_0'},
            {'kind': 'DEF_BLOCK', 'span': [2, 3], 'text': 'auto def', 'role': 'definition'},
        ],
        'answer_links': [],
    }
    nodes = [
        {
            'id': 1, 'node_type': 'source', 'word_text': 'that',
            'word_indices': [0], 'role': 'synonym', 'raw_letters': 'TEST',
            'answer_positions': [0, 1, 2, 3], 'group_id': None,
        },
        {
            'id': 2, 'node_type': 'definition', 'word_text': 'manual definition',
            'word_indices': [2], 'role': None, 'raw_letters': None,
            'answer_positions': None, 'group_id': None,
        },
    ]
    merge_manual_into_display(wfw, nodes, 'TEST')
    kinds = [b['kind'] for b in wfw['blocks']]
    assert 'SOURCE_BLOCK' in kinds, 'SOURCE_BLOCK missing'
    assert 'DEF_BLOCK' in kinds, 'Manual DEF_BLOCK missing'
    review = [b for b in wfw['blocks'] if b['kind'] == 'REVIEW_BLOCK']
    assert len(review) == 0, 'REVIEW_BLOCK not suppressed: ' + str(review)
    auto_def = [b for b in wfw['blocks'] if b.get('text') == 'auto def']
    assert len(auto_def) == 0, 'Auto DEF_BLOCK not suppressed'
    auto_src = [b for b in wfw['blocks'] if b['kind'] == 'SOURCE_BLOCK' and b.get('evidence_status') != 'manual']
    assert len(auto_src) == 0, 'Auto SOURCE_BLOCK at covered span not suppressed'
    manual_def = [b for b in wfw['blocks'] if b.get('text') == 'manual definition']
    assert len(manual_def) == 1, 'Manual DEF_BLOCK not in blocks'
    assert len(wfw['answer_links']) == 4, 'Expected 4 links for TEST, got ' + str(len(wfw['answer_links']))
    print('PASS: scenario A — suppression and DEF_BLOCK override')

    # --- Scenario B: group_id drives piece_N colour role ---
    wfw2 = {'blocks': [], 'answer_links': []}
    nodes2 = [
        {
            'id': 3, 'node_type': 'source', 'word_text': 'first',
            'word_indices': [0], 'role': 'synonym', 'raw_letters': 'TE',
            'answer_positions': [0, 1], 'group_id': 0,
        },
        {
            'id': 4, 'node_type': 'source', 'word_text': 'second',
            'word_indices': [1], 'role': 'synonym', 'raw_letters': 'ST',
            'answer_positions': [2, 3], 'group_id': 1,
        },
    ]
    merge_manual_into_display(wfw2, nodes2, 'TEST')
    src = [b for b in wfw2['blocks'] if b['kind'] == 'SOURCE_BLOCK']
    assert [b['role'] for b in src] == ['piece_0', 'piece_1'], 'Colour roles wrong: ' + str([b['role'] for b in src])
    lmap = {lk['answer_index']: lk['source_role'] for lk in wfw2['answer_links']}
    assert lmap.get(0) == 'piece_0' and lmap.get(1) == 'piece_0', 'Group 0 link roles wrong'
    assert lmap.get(2) == 'piece_1' and lmap.get(3) == 'piece_1', 'Group 1 link roles wrong'
    assert len(wfw2['answer_links']) == 4, 'Expected 4 links for TEST'
    print('PASS: scenario B — group_id drives colour roles')

    # --- Scenario C: letter validation failure --- 
    # raw_letters='WRONG' (5 chars) vs answer_positions=[0,1,2,3] (4 positions):
    # length mismatch → fails. Alternatively letters vs answer chars would also fail.
    wfw3 = {'blocks': [], 'answer_links': []}
    nodes3 = [
        {
            'id': 5, 'node_type': 'source', 'word_text': 'word',
            'word_indices': [0], 'role': 'synonym', 'raw_letters': 'WRONG',
            'answer_positions': [0, 1, 2, 3], 'group_id': None,
        },
    ]
    merge_manual_into_display(wfw3, nodes3, 'TEST')
    failed = [b for b in wfw3['blocks'] if b.get('evidence_status') == 'failed']
    assert len(failed) == 1, 'Bad letters should produce one failed SOURCE_BLOCK'
    assert len(wfw3['answer_links']) == 4, 'Expected 4 plain links for failed node, got ' + str(len(wfw3['answer_links']))
    plain_links = [lk for lk in wfw3['answer_links'] if lk['source_role'] is None]
    assert len(plain_links) == 4, 'All 4 links should be plain for failed node'
    print('PASS: scenario C — letter validation failure')

    # --- Scenario D: manual-only stub — partial evidence, complete answer row ---
    # Only positions 0-1 claimed. Positions 2-5 must appear as plain tiles.
    answer = 'HEATED'
    wfw4 = {
        'blocks': [],
        'answer_links': [],
        'clue_text': 'That man before date affected with passion?',
        'answer': answer,
        'tokens': [],
        'operations': [],
        'review_messages': ['Manual evidence only — no automatic proof'],
        'objections': [],
        'proof_source': 'manual_only',
        'proof_row_id': None,
        'missing_enrichments': [],
    }
    nodes4 = [
        {
            'id': 6, 'node_type': 'source', 'word_text': 'That man',
            'word_indices': [0, 1], 'role': 'synonym', 'raw_letters': 'HE',
            'answer_positions': [0, 1], 'group_id': 0,
        },
    ]
    merge_manual_into_display(wfw4, nodes4, answer)
    assert len(wfw4['answer_links']) == 6, 'Expected 6 links for HEATED, got ' + str(len(wfw4['answer_links']))
    lmap4 = {lk['answer_index']: lk for lk in wfw4['answer_links']}
    assert lmap4[0]['source_role'] == 'piece_0', 'Position 0 should be piece_0'
    assert lmap4[1]['source_role'] == 'piece_0', 'Position 1 should be piece_0'
    for p in [2, 3, 4, 5]:
        assert lmap4[p]['source_role'] is None, f'Position {p} should be plain'
    letters = ''.join(lmap4[p]['letter'] for p in range(6))
    assert letters == 'HEATED', 'Answer row letters wrong: ' + letters
    print('PASS: scenario D — complete answer row with partial manual evidence')

    # --- Scenario E: duplicate position claim → conflicted position is plain ---
    # Node id=7 claims H at position 0 with group_id=0.
    # Node id=8 also claims H at position 0 with group_id=1.
    # The conflict must make position 0 plain (source_role=None).
    # answer_links must have exactly 6 entries with no duplicate positions.
    wfw5 = {'blocks': [], 'answer_links': []}
    nodes5 = [
        {
            'id': 7, 'node_type': 'source', 'word_text': 'That',
            'word_indices': [0], 'role': 'synonym', 'raw_letters': 'H',
            'answer_positions': [0], 'group_id': 0,
        },
        {
            'id': 8, 'node_type': 'source', 'word_text': 'man',
            'word_indices': [1], 'role': 'synonym', 'raw_letters': 'H',
            'answer_positions': [0], 'group_id': 1,
        },
    ]
    merge_manual_into_display(wfw5, nodes5, 'HEATED')
    assert len(wfw5['answer_links']) == 6, 'Expected 6 tiles for HEATED, got ' + str(len(wfw5['answer_links']))
    positions5 = [lk['answer_index'] for lk in wfw5['answer_links']]
    assert len(positions5) == len(set(positions5)), 'Duplicate positions: ' + str(positions5)
    assert positions5 == list(range(6)), 'Positions must be 0..5: ' + str(positions5)
    lmap5 = {lk['answer_index']: lk for lk in wfw5['answer_links']}
    assert lmap5[0]['source_role'] is None, 'Conflicted position 0 must be plain'
    print('PASS: scenario E — duplicate claim yields plain tile, no duplicate positions')
    "

Expected output:

    PASS: scenario A — suppression and DEF_BLOCK override
    PASS: scenario B — group_id drives colour roles
    PASS: scenario C — letter validation failure
    PASS: scenario D — complete answer row with partial manual evidence
    PASS: scenario E — duplicate claim yields plain tile, no duplicate positions

### Check 4 — rerun safety code audit

Read lines 880-900 of web/routes/admin.py (the upfront clear block of
_rerun_clue_inner). Paste those lines verbatim and confirm:

a. The only DELETE statement references structured_explanations.
b. The only UPDATE statement references clues.
c. Neither manual_evidence_nodes nor manual_evidence_edges appears anywhere
   in those lines.

---

## After writing

Paste:
1. Lines 880-900 of web/routes/admin.py (for Check 4).
2. Output of Check 1.
3. Output of Check 2.
4. Output of Check 3.
