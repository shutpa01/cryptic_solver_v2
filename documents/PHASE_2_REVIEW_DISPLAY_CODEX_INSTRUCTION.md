# Phase 2 Review Display Fix — Codex Instruction

## Task

Improve display_from_stage_three_proof in
signature_solver/wfw_display_adapter.py so that REVIEW proofs show
the partial role information the solver found, rather than showing
every unresolved word as blank "Needs WFW role" blocks.

Only this one function changes. No other file may be touched.


---

## Problem

When a clue has a REVIEW stage_three proof, the reviewer (the site
owner, not an end user) sees the definition block plus one red
"Needs WFW role" block for every remaining word. The blocks show no
useful information. The reviewer cannot tell which words the solver
has already partially identified (as indicator candidates, separator
candidates, source candidates) and which words are completely unknown.

The proof already contains that partial information in word_purposes.
It is not being used.


---

## What word_purposes contains

proof["word_purposes"] is a list of dicts, one per clue word:

    {
        "index": 6,
        "text": "odd",
        "purpose": "operation_indicator_candidate",
        "status": "candidate",
        "evidence": [...]
    }

purpose values that appear in practice:

    "operation_indicator_candidate"     — word looks like an indicator
    "operation_indicator_modifier_candidate" — modifier of an indicator
    "structural_separator_candidate"    — joins wordplay pieces
    "definition_separator_candidate"    — separates definition from wordplay
    "unresolved_purpose"                — solver has no role for this word

status values:

    "resolved"   — fully verified
    "candidate"  — partial evidence, not yet verified
    "unresolved" — no role found at all


---

## The change

### Step 1 — build a purpose lookup inside display_from_stage_three_proof

After the line:

    context = build_wfw_atom_context(clue_text, answer)

add:

    word_purpose_by_index = {
        wp["index"]: wp
        for wp in (proof.get("word_purposes") or [])
        if wp.get("index") is not None
    }

### Step 2 — pass the lookup into _stage_three_display_block

Change the blocks loop from:

    for idx, block in enumerate(proof.get("blocks") or []):
        blocks.append(_stage_three_display_block(block, idx, answer))
    blocks.extend(_stage_three_missing_word_blocks(context, blocks))

to:

    for idx, block in enumerate(proof.get("blocks") or []):
        blocks.append(
            _stage_three_display_block(
                block, idx, answer, word_purpose_by_index))
    blocks.extend(
        _stage_three_missing_word_blocks(
            context, blocks, word_purpose_by_index))

### Step 3 — update _stage_three_display_block signature and body

Current signature:

    def _stage_three_display_block(block, idx, answer):

New signature:

    def _stage_three_display_block(block, idx, answer,
                                   word_purpose_by_index=None):

Add wp = None as the first line of the function body, before the
if kind == ... chain, so that non-review blocks do not raise
UnboundLocalError when wp is referenced in the value expression:

    kind = block.get("kind") or "REVIEW_BLOCK"
    role = block.get("role")
    wp = None                   # ← add this line
    if kind == "DEF_BLOCK":
        ...

Current REVIEW_BLOCK branch:

    elif kind == "REVIEW_BLOCK":
        role = "unaccounted"

Replace with:

    elif kind == "REVIEW_BLOCK":
        span = block.get("span")
        wp = None
        if word_purpose_by_index and span and len(span) == 2:
            for word_idx in range(span[0], span[1]):
                wp = word_purpose_by_index.get(word_idx)
                if wp:
                    break
        purpose = (wp or {}).get("purpose") or ""
        if purpose in ("operation_indicator_candidate",
                       "operation_indicator_modifier_candidate"):
            role = "review_indicator_candidate"
        elif purpose in ("structural_separator_candidate",
                         "definition_separator_candidate"):
            role = "review_separator_candidate"
        else:
            role = "unaccounted"

Also update the value field for REVIEW_BLOCKs. The current line:

    "value": answer if kind == "DEF_BLOCK" else block.get("value") or "",

Replace with:

    "value": (
        answer if kind == "DEF_BLOCK"
        else block.get("value") or _review_block_hint(wp)
    ),

Add the helper function _review_block_hint immediately before
_stage_three_display_block:

    def _review_block_hint(wp):
        """Return a short hint string for a REVIEW block, or empty string."""
        if not wp:
            return ""
        purpose = wp.get("purpose") or ""
        status = wp.get("status") or ""
        if purpose == "operation_indicator_candidate":
            return "Indicator?"
        if purpose == "operation_indicator_modifier_candidate":
            return "Indicator modifier?"
        if purpose == "structural_separator_candidate":
            return "Separator?"
        if purpose == "definition_separator_candidate":
            return "Def separator?"
        if status == "unresolved":
            return "No role found"
        return ""

### Step 4 — update _stage_three_missing_word_blocks signature

Current signature:

    def _stage_three_missing_word_blocks(context, blocks):

New signature:

    def _stage_three_missing_word_blocks(context, blocks,
                                         word_purpose_by_index=None):

In the body, for each missing token block that is appended, apply
the same purpose lookup and role assignment as in Step 3. Replace:

        missing.append({
            "block_id": "stage_three_unaccounted_%s" % token.index,
            "kind": "REVIEW_BLOCK",
            "role": "unaccounted",
            "text": token.text,
            "value": "",
            "input_value": "",
            "span": [token.index, token.index + 1],
        })

with:

        wp = (word_purpose_by_index or {}).get(token.index)
        purpose = (wp or {}).get("purpose") or ""
        if purpose in ("operation_indicator_candidate",
                       "operation_indicator_modifier_candidate"):
            missing_role = "review_indicator_candidate"
        elif purpose in ("structural_separator_candidate",
                         "definition_separator_candidate"):
            missing_role = "review_separator_candidate"
        else:
            missing_role = "unaccounted"
        missing.append({
            "block_id": "stage_three_unaccounted_%s" % token.index,
            "kind": "REVIEW_BLOCK",
            "role": missing_role,
            "text": token.text,
            "value": _review_block_hint(wp),
            "input_value": "",
            "span": [token.index, token.index + 1],
        })


---

## What the template already handles

The template atomic_parse.html already has styling for:

- role "unaccounted"        — rose/red border, "Needs WFW role" label
- role ending "_indicator"  — amber/yellow border, indicator label
- block.value non-empty     — renders as subtitle under the word text

The two new roles "review_indicator_candidate" and
"review_separator_candidate" do NOT end with "_indicator" and are
not in any existing role list. They will fall through to the default
white border in the template. That is acceptable for this fix — the
value hint text ("Indicator?", "Separator?") still appears as a
subtitle and gives the reviewer useful information.

If the template needs updating to give these new roles distinct colours,
that is a separate task. Do not touch the template in this change.


---

## What not to do

Do not modify display_from_wfw_proof_attempt.
Do not modify any other function in wfw_display_adapter.py.
Do not modify the template.
Do not modify any other file.


---

## After writing

Paste the four modified or added items in full:

- _review_block_hint (new function)
- _stage_three_display_block (modified)
- _stage_three_missing_word_blocks (modified)
- display_from_stage_three_proof (modified — blocks loop and
  word_purpose_by_index lookup only; rest unchanged)

Claude will audit before anything is run.
