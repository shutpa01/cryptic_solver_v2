# Phase 2 _blocks Defensive Access Fix — Codex Instruction

## Task

Fix an AttributeError regression in the Stage Three test suite caused by
_blocks() directly accessing casefile.operation_candidates without a
defensive fallback. The crash fires because all hand-built SimpleNamespace
test fixtures in test_stage_three_proof.py omit operation_candidates.

One file changes: signature_solver/stage_three_proof.py.
No other file changes.

Do not change test_stage_three_proof.py, stage_two_casefile.py,
wfw_display_adapter.py, or any other file.


---

## Background

The candidate preservation change (PHASE_2_CANDIDATE_EVIDENCE_PRESERVATION)
was implemented and passed focused DB verification. It then exposed a
pre-existing robustness gap: _blocks() iterates four casefile collection
attributes without getattr protection:
  casefile.definition_candidates  (line 923)
  casefile.working_pairs          (line 947)
  casefile.operation_candidates   (line 973)
  casefile.unresolved_words       (line 997)

Real StageTwoCaseFile objects always carry all four fields. Hand-built
SimpleNamespace casefiles in the test suite do not. The crash is:

    AttributeError: 'types.SimpleNamespace' object has no attribute
    'operation_candidates'

The error fires on the very first test case (clean_case) because that fixture
has definition_candidates and working_pairs but omits operation_candidates.

The candidate preservation block added in build_stage_three_proof() already
uses defensive getattr for source_candidates and operation_candidates — so
the pattern is established. _blocks() predates that discipline and has not
been updated.


---

## Change: add four defensive locals at the top of _blocks

Location: _blocks (lines 920-1004).

The function currently begins:

    def _blocks(casefile, assembly, conditional_assemblies):
        source_spans = set()
        operation_spans = set()
        for definition in casefile.definition_candidates:

Replace those first four lines with:

    def _blocks(casefile, assembly, conditional_assemblies):
        source_spans = set()
        operation_spans = set()
        _defs = tuple(getattr(casefile, "definition_candidates", ()) or ())
        _pairs = tuple(getattr(casefile, "working_pairs", ()) or ())
        _ops = tuple(getattr(casefile, "operation_candidates", ()) or ())
        _unresolved = tuple(getattr(casefile, "unresolved_words", ()) or ())
        for definition in _defs:

Then make three further substitutions, each a single identifier change:

  Line 947 — change:
      for pair in casefile.working_pairs:
  to:
      for pair in _pairs:

  Line 973 — change:
      for operation in casefile.operation_candidates:
  to:
      for operation in _ops:

  Line 997 — change:
      for item in casefile.unresolved_words:
  to:
      for item in _unresolved:

These are the only changes in the file. The body of every loop is unchanged.


---

## What not to change

Do not change _operation_check, _operation_attachment_check,
_mechanism_rules_check, or _transformations. Those functions also access
casefile.working_pairs directly, but all existing test fixtures explicitly
set working_pairs, so they do not crash and their scope is different.

Do not add defensive getattr to _definition_check or _span_integrity_check.
Both access casefile.definition_candidates directly, but all existing
fixtures set that field.

Do not add operation_candidates to any test fixture. The defensive locals
return () for missing attributes, so the tests pass without fixture changes.

Do not change the schema string "stage_three_proof:v1".

Do not change the candidate preservation block added by
PHASE_2_CANDIDATE_EVIDENCE_PRESERVATION. That block uses getattr already
and is unaffected by this change.


---

## Verification

Run two checks in order.

Check 1 — syntax only:

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        -m py_compile signature_solver\stage_three_proof.py

Expected: no output, exit 0.

Check 2 — full regression suite:

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        signature_solver\test_stage_three_proof.py

Expected output:

    Stage Three proof contract passed

All assertions must pass. Do not declare done until the exact expected
output appears.

Note on environment: the test file itself does not import Flask or admin.py.
The import chain is stage_three_proof -> stage_two_casefile -> db. No
Flask-capable environment is required for this check, so use the project
virtualenv rather than the AI_Solver Flask environment.


---

## After writing

Paste:
  1. The changed opening of _blocks (from the def line through
     "for definition in _defs:").
  2. The changed line 947 (the for pair line only).
  3. The changed line 973 (the for operation line only).
  4. The changed line 997 (the for item line only).

Then run both verification commands and paste the full output of each.
