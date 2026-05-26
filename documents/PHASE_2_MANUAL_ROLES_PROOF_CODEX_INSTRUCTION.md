# Phase 2 Manual WFW Roles as Proof Input — Codex Instruction

## Task

Make the manual word roles saved in clue_word_roles feed into the Stage Three
proof at reverify time, so that admin-assigned roles affect the verification
result rather than being display-only overlays.

Two files change: web/routes/admin.py and
signature_solver/stage_three_proof.py. No other file changes.

Do not touch Stage Two build logic, the display adapter, the template,
or the ExplanationVerifier text-based path (reverify_clue at admin.py:757).


---

## Background

The puzzle-level atomic_reverify_puzzle route (admin.py:1786) reconstructs a
casefile SimpleNamespace from stored stage_two_json, then calls
build_stage_three_proof(casefile). It never reads clue_word_roles.
Stage Three's _word_purposes builds word purposes from Stage Two blocks,
operation candidates, enrichments, and grammar alone. Words the system
could not classify remain unresolved_purpose, causing word_purpose_coverage
to REVIEW even after an admin has manually classified them.

MAUI (clue_id 10069319) illustrates the problem. The stored proof already
has answer_assembly PASS (MA + U + I = MAUI) and source_evidence PASS.
It fails on definition_evidence (no DB candidate) and word_purpose_coverage
(with, by, one, in, Hawaii are unresolved). clue_word_roles already has
the correct manual classifications: with=link, by=link, one/in/Hawaii=definition,
Graduate=synonym/MA, uniform=nato_phonetic/U, island=abbreviation_source/I.
Reverify never reads them.

Index note: clue_word_roles.word_index is confirmed to use word-sequential
indices that match Stage Three's indexing scheme (punctuation tokens are
not counted, so word_index matches the index field in word_purposes exactly).


---

## Change 1: Load manual roles in atomic_reverify_puzzle (admin.py)

Location: the for-row loop inside atomic_reverify_puzzle, immediately
after the casefile SimpleNamespace is fully constructed (the block ending
with `status=stage_two.get("status") or "unknown"`) and before the call
to build_stage_three_proof.

Add two steps:

Step A — Load raw manual roles from clue_word_roles and attach to casefile:

    manual_role_rows = db.execute(
        "SELECT word_index, word_text, role, letters "
        "FROM clue_word_roles WHERE clue_id = ? ORDER BY word_index",
        (row["id"],),
    ).fetchall()
    casefile.manual_roles = [
        {
            "index": r["word_index"],
            "text": r["word_text"],
            "role": r["role"],
            "letters": r["letters"],
        }
        for r in manual_role_rows
    ]

Step B — Build manual definition candidates (needs ref DB) and attach:

    ref_db = current_app.get_shared_ref_db()
    casefile.manual_definition_candidates = (
        _build_manual_definition_candidates(
            casefile.answer, casefile.manual_roles, ref_db)
    )

Add the following module-level helper function in admin.py (place it near
the other private helpers in that file, or directly above
atomic_reverify_puzzle):

    def _build_manual_definition_candidates(answer, manual_roles, ref_db):
        """Build definition candidates from consecutive definition manual roles.

        For each consecutive run of words manually marked 'definition',
        check whether RefDB contains that phrase -> answer. Return a list of
        candidate dicts using boundary_status 'manual_definition' (DB hit) or
        'manual_definition_gap' (missing DB fact).

        The returned list is injected into Stage Three's definition check so
        it can (a) accept the definition if the fact exists, or (b) produce a
        specific enrichment request naming the exact phrase and answer if not.
        """
        def_entries = [
            r for r in manual_roles if r.get("role") == "definition"
        ]
        if not def_entries:
            return []

        # Group consecutive word_index values into runs.
        runs = []
        current = [def_entries[0]]
        for entry in def_entries[1:]:
            if entry["index"] == current[-1]["index"] + 1:
                current.append(entry)
            else:
                runs.append(current)
                current = [entry]
        runs.append(current)

        candidates = []
        for run in runs:
            span = [run[0]["index"], run[-1]["index"] + 1]
            text = " ".join(e["text"] for e in run)
            db_hit = ref_db.is_definition_of(text, answer)
            candidates.append({
                "boundary_status": (
                    "manual_definition" if db_hit else "manual_definition_gap"
                ),
                "objections": [],
                "span": span,
                "text": text,
                "missing_answer": None if db_hit else answer,
            })
        return candidates

Existing call site: build_stage_three_proof(casefile) does not change.
build_stage_three_proof will read manual_roles and manual_definition_candidates
from casefile via getattr (see Change 2).


---

## Change 2: Consume manual roles in stage_three_proof.py

Four targeted changes inside signature_solver/stage_three_proof.py.
No imports change. No new module-level constants.


### Change 2a: build_stage_three_proof — pass extra candidates to _definition_check

Current call in build_stage_three_proof:

    checks = (
        _definition_check(casefile),
        ...
    )

Replacement — extract manual candidates from casefile and pass them:

    _manual_def_candidates = tuple(
        getattr(casefile, "manual_definition_candidates", None) or ()
    )
    checks = (
        _definition_check(casefile, _manual_def_candidates),
        ...
    )

The rest of the checks tuple is unchanged.

Also in build_stage_three_proof, pass manual_roles to _word_purposes:

Current:

    word_purposes = tuple(_word_purposes(casefile, blocks, enrichments))

Replacement:

    _manual_roles = tuple(
        getattr(casefile, "manual_roles", None) or ()
    )
    word_purposes = tuple(
        _word_purposes(casefile, blocks, enrichments, _manual_roles))


### Change 2b: _definition_check — accept extra_candidates parameter

Current signature and body (lines ~112-135):

    def _definition_check(casefile):
        accepted = [
            item for item in casefile.definition_candidates
            if _accepted_definition_candidate(item)
        ]
        if accepted:
            return StageThreeCheck(
                "definition_evidence",
                PASS,
                "accepted definition-answer evidence found",
                accepted,
            )
        if casefile.definition_candidates:
            return StageThreeCheck(
                "definition_evidence",
                REVIEW,
                "definition candidates exist but need stronger boundary evidence",
                [item for item in casefile.definition_candidates],
            )
        return StageThreeCheck(
            "definition_evidence",
            REVIEW,
            "no accepted definition-answer evidence found",
        )

Replacement:

    def _definition_check(casefile, extra_candidates=()):
        all_candidates = list(casefile.definition_candidates) + list(extra_candidates)
        accepted = [
            item for item in all_candidates
            if _accepted_definition_candidate(item)
        ]
        if accepted:
            return StageThreeCheck(
                "definition_evidence",
                PASS,
                "accepted definition-answer evidence found",
                accepted,
            )
        # Produce a precise gap message for manual_definition_gap candidates.
        gap_candidates = [
            item for item in all_candidates
            if item.get("boundary_status") == "manual_definition_gap"
        ]
        if gap_candidates:
            gaps = "; ".join(
                "%s -> %s not in DB" % (
                    item.get("text", "?"), item.get("missing_answer", "?"))
                for item in gap_candidates
            )
            return StageThreeCheck(
                "definition_evidence",
                REVIEW,
                "definition gap: %s — add to definition_answers_augmented" % gaps,
                gap_candidates,
            )
        if all_candidates:
            return StageThreeCheck(
                "definition_evidence",
                REVIEW,
                "definition candidates exist but need stronger boundary evidence",
                all_candidates,
            )
        return StageThreeCheck(
            "definition_evidence",
            REVIEW,
            "no accepted definition-answer evidence found",
        )


### Change 2c: _accepted_definition_candidate — accept manual_definition

Current (lines ~138-149):

    def _accepted_definition_candidate(item):
        if item.get("objections"):
            return False
        if (item.get("boundary_status") == "legacy_solver"
                and item.get("span_status") == "mapped"
                and item.get("span")):
            return True
        return item.get("boundary_status") in {
            "complete_edge_phrase",
            "edge_db_hit_no_larger_pos_phrase",
            "non_edge_db_hit",
        }

Replacement — add "manual_definition" to the accepted set:

    def _accepted_definition_candidate(item):
        if item.get("objections"):
            return False
        if (item.get("boundary_status") == "legacy_solver"
                and item.get("span_status") == "mapped"
                and item.get("span")):
            return True
        return item.get("boundary_status") in {
            "complete_edge_phrase",
            "edge_db_hit_no_larger_pos_phrase",
            "non_edge_db_hit",
            "manual_definition",
        }

Note: "manual_definition_gap" is deliberately NOT in the accepted set.
It must remain as a REVIEW candidate so the specific gap message fires.


### Change 2d: _word_purposes — add manual roles as final fallback

Current signature (line ~1000):

    def _word_purposes(casefile, blocks, enrichments):

New signature:

    def _word_purposes(casefile, blocks, enrichments, manual_roles=()):

Add a helper function (place immediately before _word_purposes):

    def _manual_roles_by_index(manual_roles):
        """Return a dict mapping word index to manual role entry."""
        out = {}
        for entry in (manual_roles or ()):
            idx = entry.get("index")
            if idx is not None and idx not in out:
                out[idx] = entry
        return out


    def _purpose_for_manual_role(role):
        """Map a clue_word_roles role string to a Stage Three purpose string.

        Only structural and source-type roles are mapped. If the role is not
        recognised, return None so the word falls through to unresolved_purpose.
        """
        if role in ("link", "surface", "charade_joiner"):
            return "structural_separator"
        if role == "definition":
            return "definition_phrase_member"
        if role in ("synonym", "synonym_source", "abbreviation",
                    "abbreviation_source", "nato_phonetic",
                    "literal_source", "letter_source", "roman_numeral",
                    "single_letter", "positional_source", "reversal_source",
                    "deletion_source", "hidden_source", "homophone_source"):
            return "answer_source"
        if role in ("anagram_indicator", "reversal_indicator",
                    "container_indicator", "deletion_indicator",
                    "hidden_indicator", "homophone_indicator",
                    "first_letter_indicator", "last_letter_indicator",
                    "letter_position_indicator", "alternating_indicator",
                    "parts_indicator", "positional_indicator",
                    "spoonerism_indicator"):
            return "operation_indicator"
        return None

In `_word_purposes`, add manual roles lookup after the existing
`operation_modifier_roles` setup, before the `for index, word` loop:

    manual_by_index = _manual_roles_by_index(manual_roles)

In the per-word loop, add the manual roles fallback immediately after
the grammar evidence block and before the final `if role is None:` catch-all.

Current end of per-word loop:

        if index in grammar:
            evidence.extend(grammar[index])
        if role is None:
            role = "unresolved_purpose"
            status = "unresolved"

Replacement:

        if index in grammar:
            evidence.extend(grammar[index])
        if (role is None or role == "unresolved_purpose") and index in manual_by_index:
            manual = manual_by_index[index]
            mapped = _purpose_for_manual_role(manual.get("role") or "")
            if mapped:
                role = mapped
                status = "manual"
                evidence.append({
                    "source": "manual_role",
                    "role": manual.get("role"),
                    "text": word,
                })
        if role is None:
            role = "unresolved_purpose"
            status = "unresolved"

Precedence rule: manual roles override REVIEW_BLOCK-derived unresolved_purpose
but do NOT override verified SOURCE_BLOCK, DEF_BLOCK, or OP_BLOCK assignments
(those are handled earlier in the loop with higher priority). The condition
`role is None or role == "unresolved_purpose"` enforces this.


### Change 2e: _purpose_requests — treat "manual" status like "verified"

Current (line ~636):

    def _purpose_requests(word_purposes):
        requests = []
        for item in word_purposes:
            if item.get("status") == "verified":
                continue
            ...

Replacement — also skip "manual" status:

    def _purpose_requests(word_purposes):
        requests = []
        for item in word_purposes:
            if item.get("status") in ("verified", "manual"):
                continue
            ...

Reason: words with status "manual" have been explicitly classified by a
reviewer. Generating purpose_requests for them produces noise in the
missing_enrichments panel and implies the reviewer's classification is
incomplete. It is not — the reviewer made a deliberate structural call.


---

## What not to change

Do not change build_stage_two_casefile or any Stage Two logic.
Do not change StageTwoCaseFile (the frozen dataclass) — manual_roles are
carried on the SimpleNamespace used in the reverify route only.
Do not change the reverify_clue route (admin.py:757) — that is the old
ExplanationVerifier text-based path and is out of scope.
Do not change the reverify_puzzle route (admin.py:1879) — that is the
legacy verifier path.
Do not change _stage_three_missing_word_blocks or any display adapter.
Do not change the template.
Do not add any new imports to stage_three_proof.py.


---

## Verification

After implementing, run a read-only verification script that:

1. Connects to data/clues_master.db.
2. Loads the latest stage_two_json for clue_id 10069319 (MAUI) from
   clue_pipeline_state.
3. Reconstructs a SimpleNamespace casefile from it (same as the route does).
4. Loads manual roles from clue_word_roles for clue_id 10069319.
5. Calls _build_manual_definition_candidates with the MAUI answer and those
   roles and a RefDB instance.
6. Attaches both to casefile.
7. Calls build_stage_three_proof(casefile).
8. Prints every check name, status, and detail.
9. Prints every word_purpose index, text, purpose, and status.

Expected outcome:

word_purpose_coverage:
  PASS — every clue word has a recorded purpose.
  Words Graduate, uniform, island: answer_source, status verified (from
    existing SOURCE_BLOCKs in stored proof, not from manual roles).
  Words with, by: structural_separator, status manual (from manual link roles).
  Words one, in, Hawaii: definition_phrase_member, status manual
    (from manual definition roles).

definition_evidence:
  REVIEW with message "definition gap: one in hawaii -> MAUI not in DB —
    add to definition_answers_augmented"
  (because definition_answers_augmented has "Pacific island" for MAUI,
   not "one in Hawaii")

answer_assembly:
  PASS (unchanged — MA + U + I = MAUI was already passing)

purpose_requests:
  No requests for with, by, one, in, Hawaii (their status is "manual").

If word_purpose_coverage does not PASS, print the full word_purposes list
to diagnose which words are still unresolved. Do not declare done until the
expected outcome matches.


---

## After writing

Paste:
  1. The full body of _build_manual_definition_candidates as added to admin.py.
  2. The two helper functions added to stage_three_proof.py
     (_manual_roles_by_index and _purpose_for_manual_role).
  3. The changed end-of-loop block in _word_purposes (from the grammar
     evidence extension through to the final role is None catch-all).
  4. The changed _definition_check signature and body.

Then run the verification script and paste the output.
