# Phase 2 Display Block Deduplication — Codex Instruction

## Task

Add a deduplication helper in
signature_solver/wfw_display_adapter.py so that when two display
blocks cover the same token span only the highest-priority block
is shown.

Only this one file changes. Do not touch Stage Two, Stage Three,
the template, or any other file.


---

## Background

stage_three_proof._blocks can emit both an OP_BLOCK (from
operation_candidates, e.g. reversal_indicator) and a REVIEW_BLOCK
(from unresolved_words) for the same token span. Both reach
display_from_stage_three_proof and both get rendered. The result
is the same clue word appearing twice on screen — once with a
meaningful candidate role and once as "Needs WFW role". That is
misleading noise.

Example:
    Retired    Reversal indicator
    Retired    Needs WFW role

    embracing  Container indicator
    embracing  Needs WFW role

The fix is a display-only deduplication pass that keeps only the
highest-priority block for each span and discards the rest. It does
not alter the stored proof, Stage Three status, or solver confidence.


---

## Change 1: add _dedupe_stage_three_display_blocks

Add this function immediately after _normalise_anagram_display_roles
and before display_from_stage_three_proof.

    def _dedupe_stage_three_display_blocks(blocks):
        """Remove duplicate display blocks that share the same token span.

        stage_three_proof._blocks can yield both an OP_BLOCK (from
        operation_candidates) and a REVIEW_BLOCK (from unresolved_words)
        for the same span.  When multiple blocks share a span, only the
        highest-priority one is kept.

        Priority (highest first):
            DEF_BLOCK:                       100
            SOURCE_BLOCK role=source_review:  90
            SOURCE_BLOCK (any other role):    80
            OP_BLOCK:                         70
            any role ending '_indicator':     70
            review_indicator_candidate:       60
            review_separator_candidate:       50
            LINK_BLOCK:                       30
            other:                            20
            REVIEW_BLOCK / unaccounted:       10

        Blocks with no span (None or not a two-item sequence) are always
        kept and are never candidates for deduplication.
        Mutates blocks in place.
        Display-only: does not affect proof status, solver confidence,
        or stored evidence.
        """
        def _priority(block):
            kind = block.get("kind") or ""
            role = block.get("role") or ""
            if kind == "DEF_BLOCK":
                return 100
            if kind == "SOURCE_BLOCK" and role == "source_review":
                return 90
            if kind == "SOURCE_BLOCK":
                return 80
            if kind == "OP_BLOCK":
                return 70
            if role.endswith("_indicator"):
                return 70
            if role == "review_indicator_candidate":
                return 60
            if role == "review_separator_candidate":
                return 50
            if kind == "LINK_BLOCK":
                return 30
            if role == "unaccounted":
                return 10
            return 20

        seen = {}   # span_tuple -> highest-priority block seen so far
        no_span = []
        for block in blocks:
            span = block.get("span")
            if not span or len(span) != 2:
                no_span.append(block)
                continue
            key = tuple(span)
            if key not in seen or _priority(block) > _priority(seen[key]):
                seen[key] = block

        blocks[:] = list(seen.values()) + no_span


---

## Change 2: call site in display_from_stage_three_proof

After the call to _normalise_anagram_display_roles and before
operation_detail = _stage_three_operation_detail(proof), insert
one line:

    _dedupe_stage_three_display_blocks(blocks)

The surrounding context after the change should look exactly like
this:

    _normalise_anagram_display_roles(blocks, answer_links)

    _dedupe_stage_three_display_blocks(blocks)

    operation_detail = _stage_three_operation_detail(proof)
    return {
        ...

The blocks list is sorted inside the return dict via
sorted(blocks, key=_block_sort_key). Because the deduplicator
mutates blocks in place before that sort, the updated list will
be present in the sorted output.

Do NOT insert the call before _stage_three_missing_word_blocks.
The correct position is after all blocks are assembled and after
_normalise_anagram_display_roles.


---

## What not to change

Do not change _normalise_anagram_display_roles.
Do not change _stage_three_display_block.
Do not change _stage_three_missing_word_blocks.
Do not change _stage_three_indicator_role.
Do not change any other function.
Do not touch the template.
Do not modify any other file.


---

## Note: review_separator_candidate label

The current _review_block_hint returns "Separator?" or "Def separator?"
as the tile value for review_separator_candidate blocks. A follow-up
change may update those strings (e.g. to "Joiner/Separator"). That
change is out of scope for this instruction and must not be made here.


---

## After writing

Paste the full body of _dedupe_stage_three_display_blocks and the
call-site lines (from _normalise_anagram_display_roles through to
operation_detail) so the result can be audited before anything is run.
