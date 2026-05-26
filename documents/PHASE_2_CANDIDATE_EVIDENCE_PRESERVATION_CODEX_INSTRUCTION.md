# Phase 2 Candidate Evidence Preservation — Codex Instruction

## Task

Preserve candidate source evidence in the Stage Three proof object so that
source candidates found by Stage Two and stored in stage_two_json are carried
into the Stage Three proof JSON even when no complete assembly was selected.
Also preserve operation candidates.

This is a stored-proof preservation change only. It does not change
wfw_display_adapter.py and will not make candidate evidence appear as WFW
clue-breakdown tiles. Display rendering of preserved candidates requires a
separate follow-up instruction.

One file changes: signature_solver/stage_three_proof.py.
No other file changes.

Do not change stage_two_casefile.py, wfw_display_adapter.py, admin.py,
or any other file.


---

## Background

StageTwoCaseFile already has source_candidates and operation_candidates fields.
Both are serialised into clue_pipeline_state.stage_two_json via
StageTwoCaseFile.as_dict(). When atomic_reverify_puzzle calls
_casefile_from_stage_two_json, it reads them back via
stage_two.get("source_candidates") and stage_two.get("operation_candidates").
They are therefore available to build_stage_three_proof.

The problem is that Stage Three never writes them to the returned proof object.
When no complete assembly is selected, those candidates vanish from the stored
proof entirely.

ENDEARED (10069320) is the current demonstration case. Its stored
stage_two_json contains these source candidates:
  expensive → DEAR, SYN_F, span [3,4], evidence_status verified
  energy → EN, POS_F, span [4,5], evidence_status failed
  is → DE, SYN_F, span [5,6], evidence_status verified

None of these appear in the stored Stage Three proof because no complete
answer_fit assembly was selected. They should be preserved.

Note: "fresh" Stage Two runs may find different candidates than the current
stored stage_two_json. This instruction preserves whatever is in the stored
row; improving what Stage Two finds is a separate task out of scope here.

Step 3 of the evidence trail plan (recording why _first_charade failed) is
explicitly out of scope here. It requires changes to stage_two_casefile.py
and is a separate instruction.


---

## Change 1: Extend StageThreeProof dataclass

Location: the StageThreeProof frozen dataclass (lines ~34-62).

Add two new fields at the END of the dataclass, after required_enrichments,
with empty-tuple defaults. Fields with defaults must follow fields without in
a Python dataclass.

Current end of dataclass field list:

    word_purposes: tuple[dict, ...]
    purpose_requests: tuple[dict, ...]
    unresolved_items: tuple[dict, ...]
    required_enrichments: tuple[dict, ...]

Replacement:

    word_purposes: tuple[dict, ...]
    purpose_requests: tuple[dict, ...]
    unresolved_items: tuple[dict, ...]
    required_enrichments: tuple[dict, ...]
    source_candidates: tuple[dict, ...] = ()
    operation_candidates: tuple[dict, ...] = ()

The empty-tuple default means all existing callers that do not pass these
arguments continue to work without change.


---

## Change 2: Add source_candidates and operation_candidates to as_dict

Location: StageThreeProof.as_dict (lines ~48-62).

Current end of returned dict:

        "unresolved_items": list(self.unresolved_items),
        "required_enrichments": list(self.required_enrichments),
    }

Replacement:

        "unresolved_items": list(self.unresolved_items),
        "required_enrichments": list(self.required_enrichments),
        "source_candidates": list(self.source_candidates),
        "operation_candidates": list(self.operation_candidates),
    }

The schema string "stage_three_proof:v1" does not change. The new fields are
absent in existing stored rows; any code that reads old rows must treat a
missing or null field as an empty list.


---

## Change 3: Populate in build_stage_three_proof

Location: build_stage_three_proof (lines ~65-116).

Step A — After the line that computes assembly (line ~68), insert the
following block. Use defensive getattr access for source_candidates and
operation_candidates; the reason is explained in the "What not to change"
section below.

    # Preserve Stage Two source and operation candidates in the proof.
    # Matching rule for assembly_status "used":
    #   Required: span and cleaned value must match a selected assembly part.
    #   Conditional: if BOTH the source candidate and the assembly part have
    #   a non-null token field, their tokens must also match. Token is not
    #   required when either side lacks it. Assembly parts carry a token field
    #   in current stored rows; span + value alone is therefore not always
    #   sufficient to identify the unique matching part.
    _selected_parts = []
    if assembly:
        for _part in assembly.get("parts") or []:
            _sp = _part.get("span")
            _val = _clean_answer(_part.get("value") or "")
            _tok = _part.get("token")
            if _sp and len(_sp) == 2 and _val:
                _selected_parts.append((_sp[0], _sp[1], _val, _tok))

    _source_candidates_in = tuple(
        getattr(casefile, "source_candidates", ()) or ())
    _source_candidates_out = []
    for _sc in _source_candidates_in:
        _sc_sp = _sc.get("span")
        _sc_val = _clean_answer(_sc.get("value") or "")
        _sc_tok = _sc.get("token")
        _used = False
        if _sc_sp and len(_sc_sp) == 2 and _sc_val:
            for _p0, _p1, _p_val, _p_tok in _selected_parts:
                if _sc_sp[0] != _p0 or _sc_sp[1] != _p1 or _sc_val != _p_val:
                    continue
                if _sc_tok and _p_tok and _sc_tok != _p_tok:
                    continue  # both have token but differ — not a match
                _used = True
                break
        _source_candidates_out.append(
            dict(_sc, assembly_status="used" if _used else "candidate"))
    _source_candidates_out = tuple(_source_candidates_out)
    _operation_candidates_out = tuple(
        getattr(casefile, "operation_candidates", ()) or ())

Step B — In the return StageThreeProof(...) call at the end of
build_stage_three_proof, add the two new keyword arguments after
required_enrichments:

    return StageThreeProof(
        clue_text=casefile.clue_text,
        answer=answer,
        status=status,
        checks=checks,
        blocks=blocks,
        atomic_links=links,
        transformations=transformations,
        word_purposes=word_purposes,
        purpose_requests=purpose_requests,
        unresolved_items=unresolved,
        required_enrichments=enrichments,
        source_candidates=_source_candidates_out,
        operation_candidates=_operation_candidates_out,
    )


---

## Change 4: Improve _source_check to distinguish "no candidates" from
             "candidates exist but no complete assembly was accepted"

Location: _source_check (lines ~215-261).

Current signature and first branch:

    def _source_check(casefile, assembly):
        if not assembly:
            return StageThreeCheck(
                "source_evidence",
                REVIEW,
                "no complete assembly source list to verify",
            )

Replacement of the first branch only. The rest of the function body
(from the missing = [...] line onward) is unchanged:

    def _source_check(casefile, assembly):
        if not assembly:
            _candidates = tuple(
                getattr(casefile, "source_candidates", ()) or ())
            if _candidates:
                _summary = "; ".join(
                    "%s -> %s [%s]" % (
                        c.get("text", "?"),
                        c.get("value", "?"),
                        c.get("evidence_status") or "unknown",
                    )
                    for c in _candidates[:6]
                )
                return StageThreeCheck(
                    "source_evidence",
                    REVIEW,
                    "no complete assembly found; source candidates exist but "
                    "were not accepted as a complete assembly: %s" % _summary,
                    list(_candidates),
                )
            return StageThreeCheck(
                "source_evidence",
                REVIEW,
                "no complete assembly source list to verify",
            )

The evidence_status label in brackets makes explicit that some candidates
may be "failed" and are not equally usable evidence. The cap of 6 in the
summary keeps the message readable; all candidates are still passed as the
evidence payload regardless of the cap.

The existing check logic from `missing = [...]` onward is unchanged.
This change only affects the `if not assembly:` branch.


---

## What not to change

Do not change stage_two_casefile.py or _first_charade.
Do not change wfw_display_adapter.py.
Do not change admin.py.
Do not add any new imports to stage_three_proof.py.
Do not change _assembly_check, _definition_check, _word_purposes,
_purpose_requests, or any other function.
Do not change the schema string "stage_three_proof:v1".
Scope of defensive getattr: the getattr calls on source_candidates and
operation_candidates in Change 3 prevent new AttributeError failures in
the preservation block for hand-built SimpleNamespace casefiles that omit
those fields. They do not protect _blocks, which directly accesses
casefile.operation_candidates (line ~916), as well as
casefile.definition_candidates, casefile.working_pairs, and
casefile.unresolved_words — all without getattr. Any hand-built casefile
missing those attributes will still raise AttributeError inside _blocks.
That pre-existing robustness gap is not caused by this instruction and is
out of scope here. Do not add defensive getattr to _blocks as part of
this change.


---

## Verification

After implementing, run a read-only verification script that:

Important: the script imports helpers from web.routes.admin
(_casefile_from_stage_two_json, _manual_roles_for_clue,
_build_manual_definition_candidates). Flask must be importable for those
imports to succeed. Use the Flask-capable environment:
  C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe
Do not use the project .venv or the bundled Python — both lack Flask or
markupsafe and will raise ImportError unrelated to the code change.

1. Connects to data/clues_master.db.

2. For ENDEARED (clue_id 10069320):
   a. Loads stage_two_json from clue_pipeline_state.
   b. Prints the raw stage_two_json["source_candidates"] list BEFORE any
      proof construction. This confirms what the stored row actually contains
      so the verification output is traceable.
   c. Calls _casefile_from_stage_two_json to build the casefile
      (as already implemented in admin.py).
   d. Calls build_stage_three_proof(casefile).
   e. Calls proof.as_dict().
   f. Prints every entry in proof["source_candidates"]:
      text, value, token, span, evidence_status, assembly_status.
   g. Finds the "source_evidence" check and prints its status and detail.

3. For UNDEMOCRATIC (clue_id 10069315):
   a. Same steps a and c-f.
   b. Confirms that the entries for courted, man, and in charge all appear
      in proof["source_candidates"] with assembly_status "used".
   c. Confirms that no candidate without a matching assembly part has
      assembly_status "used".

4. For MAUI (clue_id 10069319):
   a. Load manual roles from clue_word_roles using _manual_roles_for_clue.
   b. Reconstruct casefile from stage_two_json.
   c. Attach manual_roles and manual_definition_candidates using
      _build_manual_definition_candidates as already implemented in admin.py.
   d. Call build_stage_three_proof(casefile).
   e. Confirm answer_assembly still PASS (Graduate + uniform + island = MAUI).
   f. Confirm word_purpose_coverage still PASS.
   g. Print proof["source_candidates"]: Graduate, uniform, island should
      appear with assembly_status "used".

Expected results:

ENDEARED (stored stage_two_json candidates):
  Current stored candidates are:
    expensive -> DEAR, evidence_status verified
    energy -> EN, evidence_status failed
    is -> DE, evidence_status verified
  All three should appear in proof["source_candidates"] with
  assembly_status "candidate" (ENDEARED has no answer_fit assembly;
  the stored assembly has status "evidence_only" which _best_answer_fit_assembly
  does not select).

ENDEARED source_evidence check:
  Status: REVIEW.
  Detail contains: "source candidates exist but were not accepted as a
  complete assembly" and lists the three candidates with their
  evidence_status labels, for example:
    expensive -> DEAR [verified]; energy -> EN [failed]; is -> DE [verified]

UNDEMOCRATIC source_candidates:
  courted, man, in charge all present with assembly_status "used".
  No spurious "used" assignments on non-assembly candidates.

MAUI answer_assembly and word_purpose_coverage:
  Both remain PASS. No regression from the new fields.
  Graduate, uniform, island all have assembly_status "used".

If proof["source_candidates"] is empty for ENDEARED after the change, the
output from step 2b (raw stage_two_json source_candidates before proof
construction) will show whether the stored row is the cause. Do not declare
done until all three clues match the expected results.


---

## After writing

Paste:
  1. The changed end of the StageThreeProof dataclass (from word_purposes
     through the two new fields with defaults).
  2. The new annotation block in build_stage_three_proof (from
     _selected_parts = [] through _operation_candidates_out = ...).
  3. The new return statement in build_stage_three_proof showing all
     keyword arguments including source_candidates and operation_candidates.
  4. The changed first branch of _source_check (from the def line through
     the "no complete assembly source list to verify" return).

Then run the verification script and paste the full output including the
step 2b raw stage_two_json source_candidates print.
