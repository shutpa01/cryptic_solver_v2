# Phase 2 Anagram Display Roles Fix — Codex Instruction

## Task

Add a display normalisation helper in
signature_solver/wfw_display_adapter.py so that SOURCE_BLOCK fodder
pieces in anagram proofs are rendered as a unified anagram_fodder
group rather than as separate charade-coloured piece_0 / piece_1 /
piece_2 tiles.

Only this one file changes. Do not touch the template or any other file.


---

## Background

The template partials/atomic_parse.html already contains "anagram_fodder"
in its piece1_roles list, which maps to the blue colour group. No
template change is needed.

The problem is that display_from_stage_three_proof assigns SOURCE_BLOCK
fodder pieces the generic role "piece_0", "piece_1", "piece_2" etc.,
which maps them to different colour buckets (blue, pink, yellow).
For an anagram, all fodder pieces are governed by the same indicator.
They should all share the same colour to make the grouping obvious.

Changing all such roles to "anagram_fodder" achieves this.
"anagram_fodder" is already in piece1_roles so it renders as the same
blue as "piece_0". The indicator block keeps role "anagram_indicator"
and is unaffected.


---

## Change 1: add _normalise_anagram_display_roles

Add this function immediately before display_from_stage_three_proof.

    def _normalise_anagram_display_roles(blocks, answer_links):
        """For anagram proofs, normalise all SOURCE_BLOCK fodder roles to
        'anagram_fodder' so they render as a unified group rather than
        as separate charade-coloured pieces.

        Called only from display_from_stage_three_proof after blocks and
        answer_links are fully built. Mutates both lists in place.
        Display-only: does not affect proof status, solver confidence,
        or stored evidence.
        """
        # Gate on the presence of an anagram_indicator block.
        # Do not use display["operations"][0]["operation"] — that field
        # returns "anagram" for both pure anagram and anagram_charade and
        # cannot distinguish them from non-anagram proofs that happen to
        # have a block role name collision.
        has_anagram_indicator = any(
            block.get("role") == "anagram_indicator"
            for block in blocks
        )
        if not has_anagram_indicator:
            return

        # Normalise SOURCE_BLOCK fodder roles and record their spans.
        # Only blocks with a piece_N role and a recognised fodder token are
        # changed. "source_review" blocks (failed derivation evidence) are
        # not touched.
        fodder_tokens = {"ANA_F", "ABR_F", "SYN_F"}
        normalised_spans = set()
        for block in blocks:
            if block.get("kind") != "SOURCE_BLOCK":
                continue
            role = block.get("role") or ""
            if not role.startswith("piece_"):
                continue
            if block.get("token") not in fodder_tokens:
                continue
            block["role"] = "anagram_fodder"
            span = block.get("span")
            if span and len(span) == 2:
                normalised_spans.add(tuple(span))

        if not normalised_spans:
            return

        # Update answer_links whose source_span matches a normalised block.
        # Matching is by span, not by role name string. The atomic_links
        # in the proof assign source_role using the index within
        # assembly.parts ("piece_0", "piece_1", ...). The display block
        # roles use the index within the full proof blocks list, which
        # includes DEF_BLOCKs and OP_BLOCKs. These two indices are not
        # the same, so role-name matching is unreliable.
        for link in answer_links:
            span = link.get("source_span")
            if span and len(span) == 2 and tuple(span) in normalised_spans:
                link["source_role"] = "anagram_fodder"


---

## Change 2: call site in display_from_stage_three_proof

After answer_links is finalised (after the
`if answer_links: ... else: answer_links = _plain_answer_links(answer)`
block) and before `operation_detail = _stage_three_operation_detail(proof)`,
insert one line:

    _normalise_anagram_display_roles(blocks, answer_links)

The surrounding context after the change should look exactly like this:

    if answer_links:
        answer_links = sorted(answer_links, key=lambda link: link["answer_index"])
    else:
        answer_links = _plain_answer_links(answer)

    _normalise_anagram_display_roles(blocks, answer_links)

    operation_detail = _stage_three_operation_detail(proof)
    return {
        ...

The blocks list is sorted inside the return dict via
`sorted(blocks, key=_block_sort_key)`. Because the normaliser mutates
blocks in place before that sort, the updated roles will be present in
the sorted output.


---

## What not to change

Do not change display_from_wfw_proof_attempt.
Do not change _stage_three_display_block.
Do not change _stage_three_indicator_role.
Do not change any other function.
Do not touch the template.
Do not modify any other file.


---

## After writing

Paste the full body of _normalise_anagram_display_roles and the
call-site lines (from the if answer_links block through to
operation_detail) so the result can be audited before anything is run.
