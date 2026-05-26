# GT V2 Next Runbook

Date: 2026-05-17

This runbook preserves the exact next steps after the current R&D work.

## Current Position

We are in science mode, not solver-engineering mode.

The working question is:

Can grammar and structured explanations teach us block anatomy before mechanical solving?

The current answer is:

Yes, grammar adds signal, but not enough to be a single hard parse.

Therefore GT V2 should generate plausible block anatomies and let answer mechanics verify scope and assembly.

## Completed Artifacts

Core design:

- `documents/wfw_preservation_design_session_2026-05-16.md`
- `documents/gt_v2_deconstruction_experiment_plan_2026-05-17.md`
- `documents/gt_v2_block_anatomy_model_2026-05-17.md`
- `documents/operation_derived_attachment_targets_2026-05-17.md`
- `documents/gt_v2_candidate_generator_design_2026-05-17.md`

Corpus/science reports:

- `documents/structured_research_cohort_2026-05-17.md`
- `documents/clean_boundary_bucket_analysis_2026-05-17.md`
- `documents/residue_operation_signals_2026-05-17.md`
- `documents/residue_baseline_evaluation_2026-05-17.md`
- `documents/grammar_triage_targets_2026-05-17.md`
- `documents/grammar_feature_scaffold_2026-05-17.md`
- `documents/grammar_boundary_signal_2026-05-17.md`
- `documents/grammar_boundary_classifier_2026-05-17.md`
- `documents/grammar_span_boundary_experiment_2026-05-17.md`
- `documents/residue_attachment_experiment_2026-05-17.md`
- `documents/operation_attachment_slice_2026-05-17.md`
- `documents/operation_attachment_inspection_queue_2026-05-17.md`
- `documents/definition_marker_leakage_2026-05-17.md`
- `documents/block_graph_candidates_2026-05-17.md`
- `documents/block_graph_quality_2026-05-17.md`

Scripts:

- `scripts/build_structured_research_cohort.py`
- `scripts/analyze_clean_boundary_buckets.py`
- `scripts/mine_residue_operation_signals.py`
- `scripts/evaluate_residue_baseline.py`
- `scripts/mine_grammar_triage_targets.py`
- `scripts/build_grammar_feature_scaffold.py`
- `scripts/analyze_grammar_boundary_signal.py`
- `scripts/evaluate_grammar_boundary_classifier.py`
- `scripts/build_operation_attachment_slice.py`
- `scripts/build_attachment_inspection_queue.py`
- `scripts/analyze_definition_marker_leakage.py`
- `scripts/build_block_graph_candidates.py`
- `scripts/analyze_block_graph_candidates.py`

## Current Hard Results

Residue-only operation baseline:

- overall accuracy: 45%
- covered accuracy: 58%
- high-confidence accuracy: 86%

Token boundary classifier:

- lexical-only: 67%
- lexical-plus-grammar: 72%

Span boundary experiment:

- lexical exact `SOURCE` span recall: 31%
- grammar exact `SOURCE` span recall: 43%
- lexical exact `RESIDUE` span recall: 20%
- grammar exact `RESIDUE` span recall: 39%
- lexical boundary-pair accuracy: 45%
- grammar boundary-pair accuracy: 50%

Residue attachment:

- nearest-source target was invalid because it scores perfectly by construction
- parser-head attachment scored only 30% against that weak proxy
- conclusion: parser heads are not cryptic attachment

Operation attachment slice:

- residue runs: 1,144
- `UNCLASSIFIED_RESIDUE`: 480
- `OPERATOR_SCOPE`: 281
- `CONNECTOR_OR_SURFACE`: 118
- `DIRECTION_OR_ORIENTATION`: 112
- `CONTAINER_RELATION`: 100
- `ORDER_OR_POSITION`: 27
- `LOCATOR_SCOPE`: 22
- `DEF_MODIFIER`: 4

Definition marker leakage:

- 18 records have possible DBE marker leakage
- important markers: `say`, `perhaps`, `maybe`, `like`
- some are genuine homophone operators; some belong with definition

Block graph candidates:

- graph candidates: 865
- graph nodes now split mixed residue runs
- definition spans are now first-class `DEF_BLOCK` nodes
- `DEF_BLOCK -> DEFINES -> ASSEMBLY_BLOCK` is represented explicitly
- `DEF_MODIFIER_BLOCK` now attaches to `DEF_BLOCK`, not directly to the answer placeholder
- source nodes now preserve structured explanation mechanisms and values
- 846 graphs have an observed definition block
- 19 graphs have `definition_status = missing_from_structured_explanation`
- `say with` can become `say` as `DEF_MODIFIER_BLOCK` and `with` as `RELATION_BLOCK`
- this confirms that residue runs are not necessarily anatomical blocks

## Next Executable Step

When rebuilding the current artifacts, run:

```powershell
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_operation_attachment_slice.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_attachment_inspection_queue.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_block_graph_candidates.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\analyze_block_graph_candidates.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_block_type_inspection_report.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\analyze_definition_gaps.py
```

Expected outputs:

- `documents/operation_attachment_slice_2026-05-17.jsonl`
- `documents/operation_attachment_slice_2026-05-17.md`
- `documents/operation_attachment_inspection_queue_2026-05-17.md`
- `documents/block_graph_candidates_2026-05-17.jsonl`
- `documents/block_graph_candidates_2026-05-17.md`
- `documents/block_graph_quality_2026-05-17.md`
- `documents/block_type_inspection_2026-05-17.md`
- `documents/definition_gap_analysis_2026-05-17.md`

This produces weak operation-derived attachment rows and graph-shaped candidate anatomies.

## What To Inspect First

Do not inspect the whole slice as a table.

Inspect one weak label at a time.

Start with:

1. `ORDER_OR_POSITION`
2. `DIRECTION_OR_ORIENTATION`
3. `LOCATOR_SCOPE`
4. `CONTAINER_RELATION`
5. `OPERATOR_SCOPE`

The key question for each example is:

Does this residue block have the right relationship to the source block, or is it actually source-internal/surface grammar?

## What Not To Do

Do not alter `stages/`.

Do not patch the solver to fix individual examples.

Do not convert these weak labels into production truth.

Do not flatten operation plus locator into one indicator.

Do not classify connector words by word list alone.

Do not trust parser heads as cryptic attachment.

## Immediate Design Question

The next theoretical decision is whether GT V2 should represent block candidates as:

- one object per block plus relationship edges, or
- one object per complete anatomy candidate containing nested blocks

The current evidence favours both:

- block objects are easier to mine and inspect
- complete anatomy candidates are easier to verify against the answer

A sensible next design is:

- store atomic blocks
- store relationship edges
- assemble candidate anatomies as graphs
- verify graph candidates mechanically

## New Rule To Preserve

Residue runs are not blocks.

A single contiguous residue run may contain multiple anatomical blocks.

`say with` is the current exemplar:

- `say` -> `DEF_MODIFIER_BLOCK`
- `with` -> `RELATION_BLOCK`

This is structurally similar to the earlier deletion rule:

- operation and locator can preserve separately
- they still need to remain chained

Here, surface contiguity and anatomical unity are not the same thing.

## New Finding: Missing Definition Is A Signal

Do not treat absence of a `DEF_BLOCK` as harmless.

The graph run now marks this explicitly with `definition_status`.

Observed counts:

- `observed`: 846 graphs
- `missing_from_structured_explanation`: 19 graphs

The missing cases include:

- hidden clues where the full clue behaves as definition/surface
- clues where definition-by-example material such as `maybe` or `perhaps` appears without a labelled definition anchor
- terse structured explanations that mapped source/fodder but did not preserve the definition span
- letter-selection sources where the mapped source span may have swallowed locator or definition surface

This is a design issue, not a solver patch request.

GT V2 needs a way to represent:

- explicit definition block
- implicit/all-in-one definition candidate
- missing-from-explanation definition gap
- definition modifier attached to a candidate definition

The focused definition-gap report classifies all 19 gaps into useful hypotheses. The last two formerly unclassified cases became clear only after source mechanisms were preserved:

- `Muddies watercolour finally with paints` / ROILS: `Muddies watercolour` is `last_letter -> R`
- `Dance music's introduction is right for rattles` / DISCOMFITS: `Dance music's introduction is` is `last_letter -> S`

This shows why mechanism metadata belongs on `SOURCE_BLOCK`.

## Stop Point For This Branch

This branch is now at diminishing returns.

The useful design signal has been extracted:

- graph-shaped candidates are the right preservation model
- definitions and definition gaps must be explicit
- source mechanisms must be preserved
- residue runs must be splittable
- parser heads are only weak evidence

Do not keep polishing these reports unless a specific question needs answering.

The next valuable experiment should move up a level: use these lessons to design how GT V2 proposes block candidates on fresh clues without structured human explanations.

## Candidate Generator Design Draft

The stand-alone candidate generator design is now captured in:

- `documents/gt_v2_candidate_generator_design_2026-05-17.md`

It defines:

- raw-clue inputs
- candidate graph outputs
- block types and evidence rules
- generation layers
- verification rules
- failure guards
- worked SPITTOON and ARTHUR examples

The next review question is whether the example graph objects are the right shape for human inspection and later engineering.
