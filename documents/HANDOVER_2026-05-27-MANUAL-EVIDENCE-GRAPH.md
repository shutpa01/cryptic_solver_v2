# Handover — Manual Evidence Graph Instructions
# 2026-05-27

---

## What this session did

Wrote and fully reviewed two Codex implementation instruction documents for the
Manual Evidence Graph feature. Both are ready to send to Codex. Neither has been
sent yet — the user must say "send to Codex" explicitly.

There is also a pending Workstream 1 item from the previous session.

---

## Pending items (do not act without explicit instruction)

### 1. Send Slice 1 instruction to Codex (when user says so)

File: documents/PHASE_2_MANUAL_EVIDENCE_GRAPH_SLICE_1_CODEX_INSTRUCTION.md

This is the base layer. It adds:
- Two new SQLite tables (manual_evidence_nodes, manual_evidence_edges) in clues_master.db
- New module: signature_solver/manual_evidence_store.py
- Two admin routes (create node, delete node)
- Display-merge at page load for admin requests
- Admin form in clue.html
- New partial: web/templates/partials/manual_evidence_nodes.html

The document went through four rounds of user review. All issues resolved. Do not
re-review it — implement it as written.

Verification: 4 checks (syntax, temp DB round-trip, 5-scenario merge test, rerun audit).

### 2. Send Slice 2 instruction to Codex (when user says so)

File: documents/PHASE_2_MANUAL_EVIDENCE_GRAPH_SLICE_2_CODEX_INSTRUCTION.md

This extends Slice 1 to support cryptic operations. Status: revision 3.

What it adds:
- payload_json column (ALTER TABLE, migration-safe) to manual_evidence_nodes
- Two new node types: operator, transform
- Two new edge types: input_to, output_of (manual_evidence_edges table now used)
- Edge helpers: write_edge, delete_edge, get_edges_for_clue, get_graph_for_clue
- _validate_edge_types helper (also exported for testing)
- Four operation evaluators: _evaluate_reversal, _evaluate_deletion, _evaluate_anagram,
  _evaluate_container (pure functions, no DB/Flask)
- Fully replaced merge_manual_into_display with graph-aware version
- Two new admin routes (create edge, delete edge)
- Edge creation form in clue.html
- Updated partial showing nodes with prominent ids and edge list

Verification: 9 checks (syntax, temp DB round-trip, reversal merge, two-piece merge,
failed-op merge, edge type rejection, old-schema read, word_indices parsing, rerun audit).

### 3. Send PHASE_2_MANUAL_ASSEMBLY_BUILDER_CODEX_INSTRUCTION.md Rev 4 to Codex

This was approved in the session before this one. Still pending. Not related to the
manual evidence graph work.

---

## Key design decisions (do not relitigate these)

**Slice 1 decisions:**

- merge_manual_into_display builds answer_links via an authoritative position map
  (dict keyed by answer_index), not a list-append. Guarantees exactly one tile per
  cleaned-answer position. Conflicting manual claims → plain tile.
- source_block field on each answer_link is "manual_src_{id}" (not a generic string).
- answer_positions are validated against the CLEANED answer (_clean_answer strips
  non-alpha characters). Hyphens, spaces, punctuation do not count.
- group_id (0-4) drives both clue block colour and answer tile colour via piece_{N} role.
- The merge is admin-only, display-only. wfw_proof_attempts is never touched.
- Invalid non-empty answer_positions in the admin route → abort(400).

**Slice 2 decisions:**

- payload_json is added as a nullable column (ALTER TABLE, not schema change).
  get_nodes_for_clue uses PRAGMA table_info to detect whether the column exists
  before selecting it — safe against existing Slice 1 DBs.
- Transform nodes have word_indices=[]. The admin route allows empty word_indices
  only when node_type == "transform"; all other types still abort(400).
- The HTML word_indices field has had `required` removed to match (placeholder
  updated to say "blank for transform").
- json must be imported explicitly inside create_manual_evidence_node — it is not
  guaranteed to be at module level in admin.py.
- create_manual_evidence_node returns both nodes AND edges to the partial so that
  existing edges do not disappear from the HTMX list after a node is added.
- Edge type validation is in _validate_edge_types (exported from store module):
    input_to: from must be source or transform; to must be operator
    output_of: from must be transform; to must be operator
- Colour continuity: admin sets same group_id on transform as on source. Merge
  uses transform.group_id for tile colour. Source block uses source.group_id.
  Both are piece_{N} → consistent colouring across the operation.
- Failed transform: operator OP_BLOCK → evidence_status="failed"; transform
  SOURCE_BLOCK → evidence_status="failed" and role="source_review" (rose);
  input source blocks unaffected; claimed positions → plain tiles.
- Container validation: inner must appear as substring in claimed output; removing
  inner must leave exactly outer. payload_json can specify outer_node_id and
  inner_node_id; if absent, first input = outer, second = inner.

---

## File inventory

Documents written this session:
- documents/PHASE_2_MANUAL_EVIDENCE_GRAPH_SLICE_1_CODEX_INSTRUCTION.md (complete)
- documents/PHASE_2_MANUAL_EVIDENCE_GRAPH_SLICE_2_CODEX_INSTRUCTION.md (complete, rev 3)
- documents/PHASE_2_MANUAL_EVIDENCE_GRAPH_SLICE_2_SPEC_FOR_CLAUDE.md (user-written spec,
  input to the Slice 2 instruction — do not modify)

No code was written. No database was touched. No commits were made this session.

---

## What the new thread should do first

Read this handover. Then ask the user which item to action. The likely answer is
"send Slice 1 to Codex" or "send Slice 2 to Codex" — but wait for the user to say it.

Do NOT send anything to Codex without explicit "send to Codex" instruction.
Do NOT re-review the documents without being asked. They are ready.
