# GT V2 Existing System Bridge

Date: 2026-05-17

Status: design bridge from existing solver evidence

This note compares the WFW/GT V2 design to what already exists in the solver. It is not a solver patch, not a `stages/` plan, and not a proposal to replace the current DB-backed machinery.

## Bottom Line

The current solver already has most of the raw evidence GT V2 needs:

- DB-backed lexical evidence for indicators, abbreviations, synonyms, definition answers, homophones, and wordlist checks.
- Word and 2-4 word phrase analysis.
- Definition-window extraction from clue edges.
- Enhanced grammar triage already integrated before catalog solving.
- R&D graph-anatomy artifacts from structured explanations, grammar features, residue attachment, and block graph candidates.
- Flexible base-pattern matching with phrase indicators and skipped link words.
- Operation-specific answer verification for charade, container, reversal, anagram, hidden, homophone, deletion, trim, positional extraction, and combinations.

There are two different gaps:

- WFW preservation gap: current flow searches, verifies, scores, and returns one best `SignatureResult`, losing intermediate evidence and competing candidates.
- Parse coverage gap: the solver does not produce enough high-confidence parses in the first place.

The parse coverage gap is the more urgent solver problem. Atomic evidence retention is only useful to the solver if it increases the number of mechanically verified parses, or if it gives the fallback layer better constrained candidates to finish. GT V2 should therefore be evaluated as a parse-rate intervention, not only as a WFW display/preservation improvement.

Do not build a new generic semantic span system before adapting this evidence layer.

## Parse Coverage Question

The central operational question is:

Will enhanced GT V2 plus atomic evidence retention increase high-confidence parse rate?

This is not answered by the current design docs. It needs a benchmark.

Current evidence:

- Existing enhanced GT already runs before catalog solving (`signature_solver/solver.py:189`).
- The first-pass graph artifact has 865 candidate anatomies from structured explanations (`documents/block_graph_candidates_2026-05-17.md`).
- A smoke test on the first 40 graph-candidate clues found current solver high-confidence parses for 28/40 and no parse for 12/40. The sample was mostly anagram and charade: anagram 20/31 high-confidence, 11/31 unsolved; charade 8/9 high-confidence, 1/9 unsolved.
- The unsolved examples include clues whose graph artifact already has useful anatomy, such as `Man is unaccompanied when cycling` / `ELON`, `Nautical ode when reviewed having instructive value?` / `EDUCATIONAL`, and `Have a second job looming somehow by empty hut` / `MOONLIGHT`.
- The ad hoc full-cohort run was too slow to complete under a simple `solve_clue` loop, so a proper parse-rate harness needs timeouts, caching, and summarized output.

Interpretation:

The graph artifacts show plausible candidate anatomy for clues the current runtime solver may still miss. That is the exact place GT V2 could improve parse rate, but only if graph candidates are used to drive additional mechanical verification rather than merely recorded after failure.

The hypothesis should be tested in layers:

1. Baseline: current `solve_clue` high/medium/low/unsolved rates on a fixed cohort.
2. Runtime enhanced GT attribution: same cohort, but count which high-confidence solves came from `grammar_triage` versus base/positional/legacy matchers.
3. GT V2 candidate graph oracle: feed structured-explanation graph spans into mechanical verification to estimate the upper bound if span anatomy were correct.
4. GT V2 generated graph: use only fresh-clue evidence from DB, word analyzer, enhanced GT, and graph candidate rules.
5. Atomic retention retry: preserve rejected/partial source candidates and retry assemblies that the current one-best path drops.

The bridge succeeds only if steps 4 or 5 improve high-confidence parse rate over step 1 without unacceptable false positives.

## Existing Enhanced GT

There are two existing GT assets that must be preserved in the bridge:

1. `signature_solver/grammar_triage.py`: the enhanced grammar-guided solver path already called by `solve_clue`.
2. The 2026-05-17 GT V2 R&D artifacts: grammar feature scaffolds, boundary experiments, operation attachment slices, and first-pass block graph candidates.

The bridge should use both. It should not treat GT V2 as starting from a blank page.

### Runtime Enhanced GT

`signature_solver/grammar_triage.py` describes itself as grammar-guided triage that returns a `SolveResult` compatible with the existing solver pipeline (`signature_solver/grammar_triage.py:1`).

The implemented paths are:

- standalone detection for anagram, reversal, and related structural cases (`signature_solver/grammar_triage.py:7`);
- grammar signature lookup plus mechanical verification (`signature_solver/grammar_triage.py:8`);
- mechanism detection from POS patterns plus structural confirmation (`signature_solver/grammar_triage.py:9`);
- timeout-bounded fallback to `None`, so the catalog solver still runs (`signature_solver/grammar_triage.py:12`, `signature_solver/grammar_triage.py:1276`).

Important concrete behavior:

- DBE marker spans are detected as spans, not just single words; `maybe`, `perhaps`, `say`, and multi-word markers are protected from being swallowed as raw fodder (`signature_solver/grammar_triage.py:104`, `signature_solver/grammar_triage.py:139`).
- `_try_charade` builds candidate pieces from synonym, abbreviation, raw letters, first/last/outer letters, reversed synonyms, and 2-3 word phrases, then searches answer positions (`signature_solver/grammar_triage.py:455`).
- Positional letters are gated by an actual positional indicator somewhere in the wordplay window (`signature_solver/grammar_triage.py:484`).
- Reversal preserves the original synonym as metadata so the explanation does not lie about the clue source (`signature_solver/grammar_triage.py:501`).
- `grammar_triage` is already called inside `solve_clue` before catalog solving, first for each definition candidate and then as a full-clue wordplay attempt (`signature_solver/solver.py:189`, `signature_solver/solver.py:204`).

This is not just theory. It is an existing enhanced GT path that already produces `SignatureResult`/`SolveResult`. The preservation gap is that its intermediate candidates and grammar objections are still collapsed into the same final result shape as the rest of the solver.

### GT V2 R&D Artifacts

The 2026-05-17 work produced evidence that grammar is useful but not sufficient as a hard parser:

- `documents/grammar_feature_scaffold_2026-05-17.md`: 931 supervised records with source/residue boundaries and optional spaCy features.
- `documents/grammar_boundary_classifier_2026-05-17.md`: lexical token boundary accuracy 67%; lexical-plus-grammar 72%.
- `documents/grammar_span_boundary_experiment_2026-05-17.md`: exact `SOURCE` span recall improved from 31% to 43%; exact `RESIDUE` span recall improved from 21% to 39%; boundary-pair accuracy improved from 45% to 49%.
- `documents/operation_attachment_slice_2026-05-17.md`: 1,144 residue runs weakly labelled into operator scope, connector/surface, direction/orientation, container relation, order/position, locator scope, definition modifier, and unresolved residue.
- `documents/block_graph_candidates_2026-05-17.md`: 865 first-pass graph-shaped candidate anatomies from weak operation attachment labels.
- `documents/block_graph_quality_2026-05-17.md`: those 865 graphs include first-class `SOURCE_BLOCK`, `DEF_BLOCK`, `ASSEMBLY_BLOCK`, `OP_BLOCK`, `CONNECTOR_BLOCK`, `RELATION_BLOCK`, `POSITION_BLOCK`, `LOCATOR_BLOCK`, and `DEF_MODIFIER_BLOCK` nodes.

Those artifacts are not production truth, but they are existing GT V2 work. The bridge needs to connect them to the runtime solver evidence:

- use the R&D block types and edge vocabulary as the graph target;
- use the runtime DB/word analyzer/matcher as fresh-clue evidence sources;
- use enhanced `grammar_triage.py` as a high-value existing candidate generator and verifier;
- preserve failures, ambiguity, and scope objections instead of returning only one solved `SignatureResult`.

## Existing Reference Layer

`signature_solver/db.py` loads the primary reference data into memory:

- `RefDB._load_all` loads `indicators`, `wordplay`, `synonyms_pairs`, `definition_answers_augmented`, `homophones`, and builds `wordlist` (`signature_solver/db.py:30`).
- `definition_answers_augmented` is merged into `self.synonyms`, so definition answers are available through the same synonym lookup path (`signature_solver/db.py:74`).
- `get_indicator_types`, `get_abbreviations`, `get_synonyms`, `get_synonyms_substring_of`, `get_homophones`, and `is_definition_of` are the current public evidence APIs (`signature_solver/db.py:174`, `signature_solver/db.py:185`, `signature_solver/db.py:198`, `signature_solver/db.py:218`, `signature_solver/db.py:263`, `signature_solver/db.py:273`).
- `with_extra_synonyms` and `with_extra_indicators` provide read-through overlays without mutating the base DB (`signature_solver/db.py:229`, `signature_solver/db.py:246`, `signature_solver/db.py:312`, `signature_solver/db.py:403`).

Direct DB counts from this checkout:

- `data/cryptic_new.db`: `definition_answers` 263,646; `definition_answers_augmented` 644,861; `synonyms_pairs` 1,352,325; `indicators` 5,404; `wordplay` 1,592; `homophones` 1,126; `clues` 327,976.
- `data/clues_master.db`: `clues` 590,193; `structured_explanations` 183,041; `clue_word_roles` 2,100; `api_explanations` 45.
- `data/word_roles.db`: `word_roles` 65,472.

Note: the current `clues_master.db` count is 590,193 in this checkout, not 590,041.

## Existing Candidate Generation

`signature_solver/solver.py` already separates clue-level definition search from wordplay solving:

- `extract_definition_candidates` checks start and end definition windows, up to four words, against `db.is_definition_of` (`signature_solver/solver.py:87`).
- `solve_clue` normalizes the clue, extracts definition candidates, tries enhanced grammar triage for each, then falls through to catalog solving (`signature_solver/solver.py:127`).
- If grammar triage parses the full clue but a DB/Haiku definition candidate exists, the code preserves that definition on the result instead of discarding it (`signature_solver/solver.py:204`).
- `solve` analyzes the wordplay window with `analyze_phrases`, tries `match_base`, then positional matching, then legacy signature matching (`signature_solver/solver.py:535`).

`signature_solver/word_analyzer.py` already emits the span-level raw material GT V2 wants:

- `WordAnalysis.roles` stores multiple possible roles per word or phrase (`signature_solver/word_analyzer.py:16`).
- `analyze_words` adds abbreviation, indicator, synonym-substring, reverse-synonym, full-length synonym, raw/anagram/hidden/positional/deletion fodder, homophone, and link evidence (`signature_solver/word_analyzer.py:46`).
- `analyze_phrases` checks 2-, 3-, and 4-word spans for abbreviations, synonyms, and multi-word indicators (`signature_solver/word_analyzer.py:143`).

This is already a source-candidate generator. The bridge should adapt it into graph nodes rather than duplicate it.

## Existing Matching And Verification

`signature_solver/base_matcher.py` is the strongest existing bridge point:

- `match_base` precomputes per-word possible roles and word/phrase indicator maps (`signature_solver/base_matcher.py:36`).
- It places flexible `F` and `I` spans with link-word gaps (`signature_solver/base_matcher.py:186`).
- It treats multi-word phrases as possible fodder and indicator slots (`signature_solver/base_matcher.py:265`).
- `_verify_base_placement` tries possible fodder token types for each source slot, uses `_lookup_slot`, and verifies with `_verify_combo` (`signature_solver/base_matcher.py:294`).

`signature_solver/matcher.py` still carries important evidence behavior:

- `match_signatures` filters catalog entries by word count, required indicators, and letter budget before attempting assignments (`signature_solver/matcher.py:26`).
- `_try_assignments` preserves indicator candidates as either single word indices or phrase spans (`signature_solver/matcher.py:172`).
- `_assign_with_spans` places contiguous source spans and allows gaps for link/indicator words (`signature_solver/matcher.py:286`).
- `_lookup_slot` is the key evidence adapter today: it maps a clue span plus token type to possible values for anagram fodder, hidden fodder, homophones, positional extraction, synonyms, abbreviations, deletion, trim, and container outers (`signature_solver/matcher.py:368`).
- `_verify_combo` performs operation-specific answer checks and returns the assignment that currently becomes the solve (`signature_solver/matcher.py:584`).

`signature_solver/executor.py` is mostly a final formatting layer:

- `execute_signature` turns the chosen assignment into `(success, explanation, pieces)` (`signature_solver/executor.py:134`).
- `_build_explanation` attributes indicator words in human-readable text (`signature_solver/executor.py:236`).

For GT V2, executor output is too late and too flat. The graph adapter should sit before or around `_verify_combo` and `_process_match`, where placements, values, indicators, leftovers, and rejected combinations still exist.

## Where It Collapses Today

The current result object is optimized for one answer explanation:

- `SignatureResult.signature`: list of tokens.
- `SignatureResult.word_roles`: list of `(word, token, value)`.
- `SignatureResult.explanation_parts`: human-readable strings.
- `SolveResult` keeps `analyses` and `phrases`, but only one `result` survives as the main parse (`signature_solver/solver.py:47`).

This conflicts with WFW preservation:

- WFW wants original text, WFW tokens, transformations, working blocks, and answer-letter links preserved separately.
- WFW says no rich parse should be flattened into empty pieces plus a bare operation label (`documents/wfw_preservation_design_session_2026-05-16.md:529`).
- Operations and locators must stay separate but chained (`documents/wfw_preservation_design_session_2026-05-16.md:563`).
- Reversal and anagram indicators often need late binding to resolved working blocks (`documents/wfw_preservation_design_session_2026-05-16.md:581`, `documents/wfw_preservation_design_session_2026-05-16.md:597`).

The current solver can often verify the mechanics, but it does not preserve the full route.

## Mapping To GT V2 Blocks

Existing outputs map cleanly to the GT V2 candidate graph contract:

| GT V2 block | Existing source |
| --- | --- |
| `DEF_BLOCK` | `extract_definition_candidates`, `db.is_definition_of`, clue-edge windows |
| `SOURCE_BLOCK` | `WordAnalysis.roles`, phrase analyses, grammar-triage candidate pieces, `_lookup_slot` values |
| `OP_BLOCK` | indicator roles from `get_indicator_types`, `word_indicator_map`, `phrase_indicator_map`, grammar-triage structural tests |
| `LOCATOR_BLOCK` | `POS_I_*`, `DEL_I`, `_get_trim_types`, grammar-triage positional gates |
| `RELATION_BLOCK` | container indicators such as `CON_I`; relation direction resolved by `_verify_container_combo` |
| `CONNECTOR_BLOCK` | `LNK`, `db.is_link_word`, skipped leftovers, grammar/residue connector evidence |
| `DEF_MODIFIER_BLOCK` | DBE marker spans from enhanced GT and R&D residue labels |
| `SCOPE_BLOCK` | unresolved residue/scope from graph artifacts and rejected matcher placements |
| `ASSEMBLY_BLOCK` | answer value plus grammar-triage or matcher accepted operation |
| evidence | DB method/table, grammar feature, structured-explanation source, token type, span, candidate value, confidence, accepted/rejected status |

GT V2 should not emit only role labels. It should keep span, mechanism, value, source table/API, operation candidate, scope status, and verification status separate.

## Anchor Reality Checks

### SPITTOON

Clue: `Receptacle former PM brought in without delay`

Answer: `SPITTOON`

Observed DB/code facts:

- `receptacle -> SPITTOON` is absent from both `definition_answers_augmented` and `synonyms_pairs`, so `extract_definition_candidates` cannot find the clue-initial definition.
- `brought in` exists in `indicators` as `insertion`, high confidence.
- `PM -> PITT` exists via `definition_answers_augmented`, which `RefDB` merges into synonym lookup. It is not in the `wordplay` abbreviation table.
- `former PM -> PITT` is absent as a phrase.
- `without delay` has many synonym rows but no `SOON`.

Consequence:

The existing machinery has the container indicator and can use `PITT` if the span is `PM`; it lacks the phrase-level source coverage for `former PM` and `without delay -> SOON`, and lacks the definition pair `receptacle -> SPITTOON`.

Bridge interpretation:

- This is primarily a DB/source coverage gap, not a missing operation system.
- The graph must preserve `brought in` as a relation block even when source values are missing.
- It should represent `Receptacle` as a clue-initial definition candidate with missing DB support, not as no definition.
- If an overlay supplies `former PM -> PITT` and `without delay -> SOON`, existing container verification can resolve `SOON` as outer and `PITT` as inner.

### ARTHUR

Clue: `Legendary ruler in craft endlessly upset`

Answer: `ARTHUR`

Observed DB/code facts:

- `legendary ruler -> ARTHUR` is absent.
- `legendary ruler -> KING ARTHUR` is present.
- `craft -> ART` is present in `synonyms_pairs`.
- `upset -> HURT` is present via `definition_answers_augmented`.
- `endlessly` has deletion/trim evidence: `deletion/general`, `deletion/tail`, and `parts/last_delete`.
- `_get_trim_types` maps trim indicators into final-letter and related trim operations (`signature_solver/matcher.py:914`).
- `_verify_trim_charade_combo` can trim one synonym slot and charade it with the other pieces (`signature_solver/matcher.py:875`).

Consequence:

The solver already knows the operation shape needed for `ART + HUR[T] = ARTHUR`. The failure is definition exactness and candidate preservation, not lack of rule knowledge.

Bridge interpretation:

- `Legendary ruler` should become a definition candidate with partial/expanded answer evidence (`KING ARTHUR` contains `ARTHUR`), not a hard failure.
- `in` should remain available as a connector candidate between definition and wordplay, not be forced into a positional instruction.
- `endlessly` should become a locator/trim block attached to `upset -> HURT`, with the trim result preserved as working material.

## Bridge Plan

1. Build a parse-rate benchmark harness first.

   Use a fixed cohort, fixed DB, timeout per clue, and summarized output only. Report high-confidence, medium-confidence, low-confidence, unsolved, error, elapsed time, and operation mix. This is the baseline that GT V2 must beat.

2. Build a read-only evidence adapter over current solver outputs, including enhanced GT.

   Start from `solve_clue`/`solve` inputs, `grammar_triage` attempts, and `analyze_phrases` outputs. Emit candidate nodes for all word and phrase `WordAnalysis.roles`, grammar-triage candidate pieces, DBE marker spans, and structural-test evidence, with source method/table/function names where available.

3. Capture matcher attempts before collapse.

   Instrument or wrap `grammar_triage`, `match_base`, `_lookup_slot`, and `_verify_combo` to collect:

   - candidate span placement
   - source token type
   - candidate values
   - indicator span
   - DBE marker span
   - grammar/POS role sequence or structural test path
   - leftover/link spans
   - accepted verification
   - rejected verification reason when cheap to record

   Keep this read-only at first. Do not change solver selection behavior.

4. Represent definition failure explicitly.

   Use `extract_definition_candidates` as primary evidence, but also emit clue-edge definition hypotheses when DB support is missing. Mark them as `candidate_missing_db_support` or `definition_gap`, not as solved definitions.

5. Emit candidate graph objects from existing data.

   Convert existing evidence into the block vocabulary already exercised by `block_graph_candidates_2026-05-17.jsonl`:

   - `DEF_BLOCK`
   - `SOURCE_BLOCK`
   - `OP_BLOCK`
   - `LOCATOR_BLOCK`
   - `RELATION_BLOCK`
   - `CONNECTOR_BLOCK`
   - `DEF_MODIFIER_BLOCK`
   - `SCOPE_BLOCK`
   - `ASSEMBLY_BLOCK`
   - evidence edges and verification status

6. Use graph candidates to drive extra verification, then measure uplift.

   The adapter should not stop at prettier records. It should propose additional mechanical verification attempts that current one-best flow misses: alternate source spans, unresolved relation direction, operation/locator chains, and definition-gap candidates. Compare parse-rate uplift against the baseline harness.

7. Preserve overlays as evidence, not truth.

   Existing `with_extra_synonyms` and `with_extra_indicators` are useful for experiments like the SPITTOON manual overlay. Graphs should mark overlay values as overlay-sourced so DB coverage gaps stay visible.

8. Separate coverage gaps from graph gaps.

   For each failed anchor, classify the blocker:

   - missing DB pair
   - missing phrase span
   - definition exactness/containment problem
   - operation known but not preserved
   - relation direction unresolved until answer verification
   - connector/locator ambiguity

## What Not To Do

- Do not touch `stages/`.
- Do not patch solver behavior while doing this bridge work.
- Do not replace `RefDB`, `word_analyzer`, `base_matcher`, or `_lookup_slot` with a new semantic layer.
- Do not ignore `signature_solver/grammar_triage.py`; it is already an enhanced GT path in the runtime solver.
- Do not ignore the existing 2026-05-17 GT V2 artifacts; they already define and test the graph-anatomy vocabulary.
- Do not flatten GT V2 into a table of final roles.
- Do not treat missing definition evidence as harmless.
- Do not collapse operation and locator into one indicator label.
- Do not use parser-head attachment as cryptic attachment truth.
- Do not convert weak supervised labels or residue buckets into production truth.

## First Concrete Implementation Target

Create a small parse-rate harness plus a read-only graph/evidence adapter that can run on one clue and return:

- baseline solve bucket and responsible solver path where available
- original clue words and stable word spans
- definition candidates and definition gaps
- all word/phrase analyses
- enhanced-GT candidate pieces and structural test path
- base matcher placements attempted
- `_lookup_slot` values by span/token type
- accepted/rejected operation verification summaries
- final graph candidate(s) if any

The adapter should prove itself first on the two anchors above, then on the 40-clue smoke-test slice from `block_graph_candidates_2026-05-17.jsonl`. It should show whether SPITTOON and ARTHUR are blocked by coverage, representation, or search-order collapse, and whether graph-driven retries recover any of the 12/40 current unsolved cases without adding false high-confidence parses.
