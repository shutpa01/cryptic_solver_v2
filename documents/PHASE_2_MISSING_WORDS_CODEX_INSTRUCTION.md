# Phase 2 Missing Clue Words — Codex Instruction

## Task

Fix two related bugs in signature_solver/wfw_display_adapter.py that
together cause clue words to be invisible on the WFW page.

Only wfw_display_adapter.py changes. Do not touch Stage Two, Stage Three,
the template, or any other file.


---

## Context

The WFW page for DM 17885 is missing clue words across many clues.
Survey of all 30 clues in that puzzle: 15 have one or more tokens
with no display block, and word_purposes is non-empty for every single
one of them (it is never empty in practice).

Worst cases: NASCENT (6 of 9 words missing), MAUI (5 of 8 missing),
CHARLIE (5 of 7 missing), HACIENDA (5 of 10 missing).


---

## How blocks are generated

display_from_stage_three_proof iterates proof["blocks"] to build display
blocks. Stage Three puts tokens into proof["blocks"] in three ways:

  OP_BLOCK  — token is a recognised operation indicator
  DEF_BLOCK — token is part of the definition phrase
  REVIEW_BLOCK — token appears in unresolved_items (a field populated from
                 Stage Two's unresolved_words list)

Tokens that Stage Three assigned a word_purpose entry to but did NOT
include in unresolved_items end up with no block at all. These are common
— for example, NASCENT's wordplay words "In", "turn", "a", "new",
"perfume", "is" all have word_purposes entries (unresolved_purpose) but
are absent from unresolved_items, so they have no block.

The function _stage_three_missing_word_blocks exists precisely to catch
these gaps: it computes which tokens have no block and adds a REVIEW_BLOCK
for each one.


---

## Bug 1 — the guard that prevents _stage_three_missing_word_blocks
            from ever running

File: signature_solver/wfw_display_adapter.py
Function: display_from_stage_three_proof
Lines (approximate): the block after iterating proof["blocks"]

Current code:

    blocks = []
    for idx, block in enumerate(proof.get("blocks") or []):
        blocks.append(
            _stage_three_display_block(
                block, idx, answer, word_purpose_by_index))
    if not word_purpose_by_index:
        blocks.extend(
            _stage_three_missing_word_blocks(
                context, blocks, word_purpose_by_index))

The condition `if not word_purpose_by_index` means "only run the fallback
when there are no word purposes at all." But Stage Three always populates
word_purposes for every Stage One word token (punctuation tokens are not
included in word_purposes), so word_purpose_by_index is always a non-empty
dict for any clue with at least one word, always truthy. The condition is
never True. The fallback never runs.

Fix: remove the guard entirely. Call unconditionally.

    blocks = []
    for idx, block in enumerate(proof.get("blocks") or []):
        blocks.append(
            _stage_three_display_block(
                block, idx, answer, word_purpose_by_index))
    blocks.extend(
        _stage_three_missing_word_blocks(
            context, blocks, word_purpose_by_index))

_stage_three_missing_word_blocks already handles the no-gap case safely:
it computes which context token indices are already covered by existing
blocks and only adds blocks for uncovered ones. Calling it when nothing
is missing returns an empty list. Calling it unconditionally is safe.


---

## Bug 2 — index mismatch for words after mid-clue punctuation

File: signature_solver/wfw_display_adapter.py
Function: _stage_three_missing_word_blocks

This bug causes incorrect results specifically when a clue has a
punctuation token (comma, question mark) in the middle of the sentence,
and there are both covered words and uncovered words after that punctuation.

Root cause: two different indexing schemes are in use.

  Proof scheme (word-sequential): Stage Three numbers only word tokens
  with consecutive integers, skipping punctuation positions.
  Example: "In turn, a new perfume is starting to develop"
    proof index 0='In', 1='turn', 2='a', 3='new', 4='perfume',
              5='is', 6='starting', 7='to', 8='develop'

  Context scheme (all-token): build_wfw_atom_context numbers every token
  including punctuation.
  Example: same clue
    context index 0='In', 1='turn', 2=',' (punct), 3='a', 4='new',
              5='perfume', 6='is', 7='starting', 8='to', 9='develop'

After the comma, every context index is one higher than the corresponding
proof word-sequential index.

_stage_three_missing_word_blocks currently:
  - Iterates context.clue_tokens using context indices
  - Checks those context indices against `covered`, which was built from
    proof block spans (proof word-sequential indices)
  - Looks up word_purpose_by_index using context indices, when that dict
    is keyed by proof word-sequential indices

Consequence for "In turn, a new perfume is starting to develop" (NASCENT):

  The DEF_BLOCK has proof span [6, 9], so covered = {6, 7, 8}.
  At context index 6 is 'is' (word-sequential index 5).
  At context index 9 is 'develop' (word-sequential index 8).

  The function checks: is context_idx 6 in covered? Yes (because covered
  has 6 from the DEF_BLOCK, even though proof index 6 refers to 'starting'
  not 'is'). Result: 'is' is incorrectly skipped — it remains invisible.

  The function checks: is context_idx 9 in covered? No (covered stops at
  8). Result: 'develop' is incorrectly added as a missing REVIEW_BLOCK,
  even though it is already inside the DEF_BLOCK "starting to develop".

For clues where all covered tokens are BEFORE any punctuation, or where
the punctuation is at the end of the clue (e.g. a trailing '?'), there is
no drift and Bug 1's fix is sufficient.

Fix: compute a word-sequential index for each context token, then use that
for the covered check, the word_purpose_by_index lookup, and the output
block span.

Current _stage_three_missing_word_blocks:

    def _stage_three_missing_word_blocks(context, blocks,
                                         word_purpose_by_index=None):
        covered = set()
        for block in blocks:
            span = block.get("span")
            if not span or len(span) != 2:
                continue
            covered.update(range(span[0], span[1]))
        missing = []
        for token in context.clue_tokens:
            if token.kind != "word" or token.index in covered:
                continue
            wp = (word_purpose_by_index or {}).get(token.index)
            ...
            missing.append({
                ...
                "span": [token.index, token.index + 1],
            })
        return missing

Replacement:

    def _stage_three_missing_word_blocks(context, blocks,
                                         word_purpose_by_index=None):
        covered = set()
        for block in blocks:
            span = block.get("span")
            if not span or len(span) != 2:
                continue
            covered.update(range(span[0], span[1]))

        # Build a word-sequential index for each context token that matches
        # the proof's indexing scheme, which numbers only word tokens.
        # After any mid-clue punctuation token the two schemes diverge:
        # context indices include the punctuation position, proof indices skip
        # it. Using proof-equivalent word-sequential indices keeps the covered
        # check, the word_purpose lookup, and the output span consistent with
        # the proof blocks already in the list.
        word_seq = {}
        seq = 0
        for token in context.clue_tokens:
            if token.kind == "word":
                word_seq[token.index] = seq
                seq += 1

        missing = []
        for token in context.clue_tokens:
            if token.kind != "word":
                continue
            ws = word_seq.get(token.index, token.index)
            if ws in covered:
                continue
            wp = (word_purpose_by_index or {}).get(ws)
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
                "block_id": "stage_three_unaccounted_%s" % ws,
                "kind": "REVIEW_BLOCK",
                "role": missing_role,
                "text": token.text,
                "value": _review_block_hint(wp),
                "input_value": "",
                "span": [ws, ws + 1],
            })
        return missing

Note: the block_id uses ws (word-sequential) instead of token.index so
it remains consistent with the proof's indexing convention. The span also
uses ws for the same reason.


---

## What not to change

Do not change _stage_three_display_block.
Do not change _normalise_anagram_display_roles.
Do not change _dedupe_stage_three_display_blocks.
Do not change any other function.
Do not touch the template.
Do not modify any other file.


---

## Verification

The verification target is the latest persisted Stage Three proof rows
stored in wfw_proof_attempts, as rendered by display_from_stage_three_proof
applied to those rows. It is NOT a fresh live rebuild via build_stage_two_casefile
or build_stage_three_proof — a fresh rebuild can produce additional
operation/source blocks for NASCENT and CHARLIE that are not in the
stored rows. The stored rows are what the clue page actually displays.

After making both changes, run a read-only verification that calls
display_from_stage_three_proof on the latest stored proof_json for each
of the three clue_ids below. Print the resulting block list (kind, role,
text, span) and confirm the counts match.

Expected results (from latest stored wfw_proof_attempts rows):

NASCENT (clue_id 10069317):
  7 blocks total.
    - DEF_BLOCK "starting to develop"
    - REVIEW_BLOCK: "In", "turn", "a", "new", "perfume", "is"
  "develop" must NOT appear as a separate REVIEW_BLOCK tile.

CHARLIE (clue_id 10069328):
  6 blocks total (7 clue words, but definition is one two-word tile).
    - DEF_BLOCK "Silly person"
    - REVIEW_BLOCK: "shows", "daily", "piece", "of", "dishonesty"

MAUI (clue_id 10069319):
  8 blocks total.
    - SOURCE_BLOCK: "Graduate", "uniform", "island,"
    - REVIEW_BLOCK: "with", "by", "one", "in", "Hawaii"

Detailed notes:

NASCENT (clue_id 10069317):
  Clue: "In turn, a new perfume is starting to develop"
  Stored proof has one block: DEF_BLOCK [6,9] "starting to develop".
  Both Bug 1 and Bug 2 apply (comma after "turn" creates index drift).
  The six wordplay words are absent from proof["blocks"] entirely (Bug 1).
  Without Bug 2's fix, 'is' would still be invisible (context idx 6 falls
  inside covered={6,7,8}) and 'develop' would appear as a spurious duplicate
  REVIEW_BLOCK (context idx 9 falls outside covered). Both bugs must be
  fixed for the correct 7-block result.

CHARLIE (clue_id 10069328):
  Clue: "Silly person shows daily piece of dishonesty"
  Stored proof has one block: DEF_BLOCK [0,2] "Silly person".
  No mid-clue punctuation, so Bug 2 does not apply.
  Clean test for Bug 1's fix only. Five wordplay words are absent from
  proof["blocks"] and unresolved_items; they should all appear after fix.

MAUI (clue_id 10069319):
  Clue: "Graduate with uniform by island, one in Hawaii"
  Stored proof has three SOURCE_BLOCKs at spans [0,1], [2,3], [4,5].
  The comma after "island" is mid-clue punctuation. All three covered
  SOURCE_BLOCK tokens are BEFORE the comma, so Bug 2's index drift only
  affects the uncovered words after the comma ("one", "in", "Hawaii").
  Without Bug 2's fix those three words still appear (their context indices
  fall outside the covered set regardless of drift), but with wrong
  word-sequential spans. With Bug 2's fix their spans are consistent with
  the proof's indexing convention.


---

## After writing

First, paste:
  1. The full body of _stage_three_missing_word_blocks as changed.
  2. The block in display_from_stage_three_proof showing the unconditional
     call site (from the `blocks = []` line through to the
     _stage_three_missing_word_blocks call).

Do not paste the entire file.

Then run a read-only verification script (no writes) that:
  - Connects to data/clues_master.db
  - For each of clue_ids 10069317, 10069328, 10069319: loads the latest
    proof_json from wfw_proof_attempts (ORDER BY id DESC LIMIT 1)
  - Calls display_from_stage_three_proof on that proof_json
  - Prints: clue_id, answer, total block count, and for each block:
    kind | role | text | span

Confirm that actual output matches the expected block counts and contents
in the Expected results section above. If any mismatch, report it before
declaring done.
