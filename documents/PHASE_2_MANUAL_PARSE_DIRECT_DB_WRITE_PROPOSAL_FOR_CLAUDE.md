# Phase 2 Manual Parse Direct DB Write Proposal for Claude

## Problem

At the moment, saving a manual structured parse can still produce a second human-review step.

The relevant path is in `web/routes/admin.py`:

- `save_container_structured_parse`
- `save_source_structured_parse`
- `save_single_operation_structured_parse`

Each route writes the manual parse, then calls `_apply_structured_parse_score_if_valid`.

Inside `_apply_structured_parse_score_if_valid`, the parse is structurally validated, then `_audit_structured_parse_db_facts(..., queue_missing=True)` checks the reference DB. If a synonym, abbreviation, indicator, or definition from the manual parse is missing, it queues a row in `pending_enrichments` and marks the clue as:

```text
manual_structured_parse_needs_db
```

That is wrong for this workflow. The human has already reviewed the relationship in the manual parser. Requiring the same human to approve the same fact again in the enrichment dashboard is duplicative and makes the manual parser feel as though it did not really save a complete parse.

## Desired Behaviour

When a user saves a structurally valid manual structured parse:

1. Extract the DB facts implied by the parse.
2. Insert any missing facts directly into `data/cryptic_new.db`.
3. Do not create `pending_enrichments` rows for those facts.
4. Re-audit after the direct insert.
5. If the facts now exist, score the clue as `manual_structured_parse` with confidence `1.0`.
6. Only use `manual_structured_parse_needs_db` if a fact could not be normalized or inserted and still does not exist after the write attempt.

This is specifically for manual structured parse saves. It should not change the normal solver/dashboard enrichment workflow.

## Important Existing Helpers

Do not write a new parallel DB insertion implementation unless there is a strong reason.

`web/routes/admin.py` already has:

```python
def _structured_parse_db_entries(parse_dict):
    ...
```

This extracts the facts from the manual parse. It already handles reversal/anagram correctly by using operation `fodder` for the piece DB entry:

```python
piece_fodder.get(pid) or piece.get("letters")
```

So for a reversal like `family -> KIN -> NIK`, the DB fact remains `family -> KIN`, not `family -> NIK`.

`web/routes/admin.py` also already has:

```python
def _write_wfw_db_entries(entries):
    ...
```

This normalizes entries with `_normalise_wfw_db_entry`, checks `_wfw_db_entry_exists`, inserts with `_insert_wfw_db_entry`, commits, and returns `(added, existing)`.

That is the right writer to reuse.

## Proposed Code Change

File: `web/routes/admin.py`

### 1. Add a manual structured parse DB writer helper

Add a helper near `_audit_structured_parse_db_facts`:

```python
def _write_structured_parse_db_facts(parse_dict):
    entries = _structured_parse_db_entries(parse_dict)
    return _write_wfw_db_entries(entries)
```

This keeps the source of truth for extraction in `_structured_parse_db_entries` and the source of truth for insertion in `_write_wfw_db_entries`.

If we want clearer source labels than `admin_wfw_fact`, modify `_write_wfw_db_entries` to accept an optional source label, but that is not required for this slice. If changing source labels, keep the existing default unchanged for other callers.

Example optional shape:

```python
def _write_wfw_db_entries(entries, source="admin_wfw_fact"):
    ...
```

Then pass `source` into `_insert_wfw_db_entry`, which would also need an optional source parameter. This is acceptable but not necessary.

Recommendation for the narrow slice: keep the existing `admin_wfw_fact` source and avoid widening the patch.

### 2. Change `_apply_structured_parse_score_if_valid`

Current logic:

```python
audit = _audit_structured_parse_db_facts(
    db, clue, parse_dict, queue_missing=True)
if audit["missing"]:
    ...
    return {"ok": False, ...}
```

Replace with this sequence:

```python
audit = _audit_structured_parse_db_facts(
    db, clue, parse_dict, queue_missing=False)
if audit["missing"]:
    added, existing = _write_structured_parse_db_facts(parse_dict)
    audit = _audit_structured_parse_db_facts(
        db, clue, parse_dict, queue_missing=False)
    if audit["missing"]:
        ...
        return {
            "ok": False,
            "errors": errors,
            "missing": audit["missing"],
            "added": added,
            "existing": existing,
        }
```

The important details:

- `queue_missing=False` both before and after the direct write.
- No call to `_queue_pending_enrichment` for manual structured parse facts.
- Re-audit after writing rather than assuming the insert worked.
- Preserve the existing high-score path once no missing facts remain.
- The low-score fallback remains only for genuine write/audit failure.

### 3. Error text

If the post-write audit still has missing facts, the returned errors should say something like:

```text
manual parse DB write failed: synonym pound -> BEAT
```

Do not say "queued for review" because this workflow should not queue.

## What Not To Change

Do not change:

- `pending_enrichments` behaviour for solver-generated enrichment.
- Stage Two.
- Stage Three.
- WFW display.
- Manual parse validation rules.
- The manual parser UI.
- Existing dashboard "Add to DB" behaviour.
- Stored parse JSON format.

This is a save/scoring path change only.

## Verification Plan

Use direct tests or small in-memory route tests. Do not mutate live clue data as the primary verification.

### Test 1: direct write helper

Given a parse with:

- definition `Bohemian -> BEATNIK`
- piece `Pound -> BEAT` relationship `synonym`
- piece `family -> KIN` relationship `synonym`
- operation `taken -> container`

Assert `_structured_parse_db_entries` emits:

```python
{"type": "definition", "word": "Bohemian", "value": "BEATNIK"}
{"type": "synonym", "word": "Pound", "value": "BEAT"}
{"type": "synonym", "word": "family", "value": "KIN"}
{"type": "indicator", "word": "taken", "value": "container"}
```

For reversal/anagram, also assert the source fact uses `operation.fodder`, not the final answer letters.

### Test 2: no pending enrichment from manual parse

Use a temporary/admin test DB for clue metadata and either a temporary copy of `cryptic_new.db` or monkeypatch the write/existence helpers.

Flow:

1. Save a valid manual structured parse with facts deliberately absent from the reference DB.
2. Run `_apply_structured_parse_score_if_valid`.
3. Assert:
   - return `ok is True`
   - `structured_explanations.model_version == "manual_structured_parse"`
   - `confidence == 1.0`
   - no matching rows were added to `pending_enrichments`
   - the expected direct DB writer was called for the missing entries

### Test 3: still fails if direct write cannot satisfy audit

Monkeypatch `_write_structured_parse_db_facts` to do nothing and `_wfw_db_entry_exists` to continue returning false.

Assert:

- `ok is False`
- model version becomes `manual_structured_parse_needs_db`
- error text says the manual DB write failed or still missing
- no pending enrichment row is created

### Manual Smoke Test

After implementation:

1. Restart Flask if needed.
2. Open a clue with a valid manual structured parse.
3. Save the parse with a new synonym/indicator/definition that is not already in the DB.
4. Confirm the parse scores high immediately.
5. Confirm the fact appears in the relevant reference DB table.
6. Confirm the fact does not appear in pending enrichment.
7. Run puzzle-level reverify and confirm the manual parse remains high.

## Acceptance Criteria

- A valid manual structured parse is a complete human assertion.
- Missing facts from that parse are written directly to the reference DB.
- No duplicate human-review queue is created.
- Reverify keeps manual structured parses high after direct DB insertion.
- Existing non-manual enrichment flows are untouched.

