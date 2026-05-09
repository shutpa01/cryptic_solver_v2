# Catalog seeding from structured explanations — design

**Status:** draft for review (2026-05-06).
**Sits alongside:** `PARALLEL_SYSTEM_DESIGN.md` (the master architecture).
**Relationship to the cascade:** this is a one-off **seeding pass**,
not a runtime phase. Run it once across the corpus of clues with
structured explanations; the catalog grows; the runtime cascade then
benefits from the enriched catalog. Subsequent leftover processing
can be re-fed into the seeder periodically.

-----

## 1. Purpose

The leftover-processing discipline (per `feedback_leftover_process.md`)
has, over many sessions, produced a large corpus of structured
explanations in `clues_master.db.structured_explanations`. Each row
has:

- a `components` JSON describing the wordplay pieces and assembly
- a definition (`definition_text`)
- an indicator-annotated explanation
- a wordplay_type
- a confidence

These are **already** definition-checked, indicator-annotated, and
mechanism-confirmed. They cover the harder, more diverse clues the
automated phases miss — exactly the templates our matcher would
otherwise struggle to reach.

This seeding pass:

1. Reads each row's components JSON
2. Translates it into our Form schema
3. Runs clipboard_verifier against it
4. Writes PASS forms into `shadow_db.solves`/`solve_assignments`
5. Logs FAILs for human review with full diagnostic detail
6. After a batch, re-runs `extract_catalog.py` so the catalog grows
   with whatever new templates emerged

Net effect: the catalog jumps from "33 templates we extracted from a
narrow corpus" toward "every distinct shape ever manually verified
across every leftover-processed puzzle". Much wider structural
vocabulary, especially for compound mechanisms.

-----

## 2. Source corpus — TWO sources

The seeder has two complementary input sources. Each contributes
candidate Forms; both feed into the same verify-save-eliminate loop.

### Source A — structured_explanations JSON

Every row in `clues_master.db.structured_explanations` joined to
`clues` for the clue text and answer.

```sql
SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.direction,
       c.clue_text, c.answer, c.enumeration, c.explanation AS blog,
       se.components, se.wordplay_types, se.definition_text,
       se.confidence, se.model_version
FROM clues c
JOIN structured_explanations se ON se.clue_id = c.id
WHERE c.answer IS NOT NULL AND c.answer != ''
  AND se.components IS NOT NULL AND se.components != '';
```

No additional filter on confidence, model_version, or source is
applied. The clipboard verifier is the only quality gate. The
PASS/FAIL ratio across model_versions will itself be useful data.

For first batches we may want to **stratify by model_version** (e.g.
process 20 `claude_review` rows first as a warm-up, then 20
`mechanical_v1`, etc.) so we can see translator behaviour on each
class before scaling. Decision deferred to operational note (see
section 9).

### Source B — raw blog text in `clues.explanation`

Every clue with a populated `explanation` field. Source mapping:

- **TFTT** for Times
- **15x15** for Guardian and Independent

Production already populates `clues.explanation` for puzzles where
the blog has been scraped. A blog parser reads this prose, identifies
pieces / mechanism / indicator / definition (the same activity a
human does in the leftover-processing discipline), and builds a
Form per our schema.

We have prior prototype infrastructure (`blog_parser.py`,
`db_anchored_mapper.py`, `assembly_enumerator.py`) that produced the
original 61 Forms / 33 templates in `catalog_v1.json`. That pipeline
is reusable as the starting point but needs harmonising with current
discipline:

- Tokenisation must use `surface.tokenize` (apostrophe-preserving)
  — the existing prototype was written before that fix.
- Output must include indicator words explicitly (the existing
  prototype's mapping captures these but the assembly may drop them).
- Verification is via `clipboard_verifier`, not the older verifier
  the prototype was originally checked against.

### Why both sources

- Source A is **convenient** (pre-parsed JSON) but incomplete:
  ~1,700 of the highest-quality `claude_review` rows have empty
  `ai_pieces`, with the actual mechanism in the blog text and not
  in the JSON.
- Source B is **rich** but heavier to parse. It is the only path to
  compound mechanisms in their natural form.
- Production phases like `parser_fifteensquared_v1` and `tftt+haiku`
  already convert some blogs to structured_explanations, so there is
  overlap. The seeder is robust to this — duplicate Forms produced
  from the same clue across sources are deduplicated by (clue_id,
  signature) at the shadow_db.solves level.

-----

## 3. Translators (two, one per source)

### 3a. JSON translator (Source A — structured_explanations)

#### 3.1 Input shape — `components` JSON

Production writes `components` as a JSON object. Across phases the
shape varies but consistently includes:

- `ai_pieces` — list of pieces, each shaped:
  ```
  {"clue_word": "...", "letters": "...", "mechanism": "..."}
  ```
  where `mechanism` is one of:
  `synonym | abbreviation | literal | anagram_fodder | hidden |
  spoonerism | reversal | positional` (or a sub-typed variant).
- `assembly` — op-specific data:
  - hidden:    `{op: "hidden"|"hidden_reversed", words: "...", _definition: "..."}`
  - spoonerism: `{word1, word2, swapped1, swapped2, clue_word1?, clue_word2?}`
  - DD:        `{op: "double_definition", left_def, right_def}`
  - anagram:   `{op: "anagram", fodder_words: [...], indicator?}`
  - charade/etc.: `{op: "<type>", pieces: [...]}` (often)
- `wordplay_type` — single string indicating the top-level op.

The translator's job is to read these fields, figure out the
top-level op + children + leaves + indicator, and emit a Form per
`schema.py`.

### 3.2 Per-op translation rules

For each top-level wordplay_type, the translator builds a tree:

| wordplay_type | Form structure |
|---|---|
| `hidden` | `hidden(literal,…)` over the spanning words; indicator extracted from clue (see 3.4) |
| `hidden_reversed` | `reversal(hidden(literal,…))`; indicator on the hidden node |
| `spoonerism` | `spoonerism(synonym(cw1→sw1), synonym(cw2→sw2))`; indicator = "Spooner"/"Spooner's" |
| `double_definition` | `double_definition(synonym(left→ans), synonym(right→ans))`; no indicator |
| `anagram` | `anagram(literal,…)` over fodder_words; indicator extracted |
| `charade` | `charade(<piece nodes>)` from `ai_pieces`; no top-level indicator |
| `container` | `container(outer, inner)`; outer/inner identified from pieces; indicator extracted |
| `reversal` | `reversal(<child>)`; indicator extracted |
| `deletion` | `deletion(<source>, kind)`; kind from sub-type; indicator extracted |
| `acrostic` | `acrostic(literal,…, kind=first|last)`; indicator extracted |
| `homophone` | `homophone(<child>)`; indicator extracted |

Compound mechanisms (charade-with-deletion, container-of-anagram,
etc.) appear in the components as nested structures or multiple
pieces with mixed mechanisms. The translator handles these by
recursing — each piece becomes a sub-tree.

### 3.3 Per-piece leaf construction

Each `ai_pieces` entry maps to a leaf (or wrapped leaf):

| `mechanism` | Leaf built | Notes |
|---|---|---|
| `synonym` | `synonym(source_word=clue_word, value=letters)` |
| `abbreviation` | `abbreviation(source_word=clue_word, value=letters)` |
| `literal` | `literal(source_word=clue_word, value=letters)` | Letters-only of clue_word |
| `anagram_fodder` | `literal(source_word=clue_word, value=letters)` | Anagram fodder is always literal |
| `hidden` | `literal(source_word=clue_word, value=letters)` | Hidden fodder is literal |
| `positional` | `positional(source_word, value, kind)` | kind from sub-type or detected from value-vs-source |
| `homophone` | `homophone_leaf(source_word, value)` |

If a piece carries deletion sub-info (curly-brace style or explicit
deletion fields), wrap the inner literal in a `deletion(...)` node.

### 3.4 Indicator extraction

For ops that require an indicator, the translator scans the clue
text minus the definition phrase minus the leaf source words for a
clue word (or contiguous phrase) that:

1. Appears in the indicators DB
2. With `wordplay_type` matching the op (hidden, anagram, etc.)

If multiple candidates, pick the one closest in position to the
op's leaves. If none match, the translator records this as a
**translation gap** and the resulting Form will FAIL clipboard's
indicator check — which is honest information, not silent failure.

### 3.5 Definition + link words

- Definition: `definition_text` from the row, or fallback to the
  blog `_definition` field if present.
- Link words: clue tokens (per `surface.tokenize`) minus
  definition-tokens minus all leaf source_word tokens minus all
  indicator tokens. Whatever remains is the link_words list.

If any leftover token isn't on `clipboard_verifier.LINK_WORDS`, the
Form will FAIL the residue check — again, honest signal.

### 3.6 Failure modes during translation

The translator distinguishes between:

- **Translation error** — the JSON shape is malformed, an op is
  unrecognised, or required fields are missing. No Form produced.
  Logged as `translation_error`.
- **Translated but incomplete** — a Form is built but some piece is
  unaccounted (e.g., indicator not findable in DB). Form produced
  with whatever's known; the verifier fails it for the actual
  reason.

Translation errors are kept separate from verifier failures — they
reveal translator bugs, not verifier limitations.

### 3b. Blog translator (Source B — clues.explanation)

For clues with a populated `clues.explanation`, a blog parser reads
the prose and constructs a Form. The discipline mirrors what a human
does during leftover processing (per `feedback_leftover_process.md`):

1. Identify the **wordplay structure** from the prose. Bloggers
   typically use:
   - Curly-brace deletion notation: `REN{t}`, `{cha}RT{ed}`
   - Parens-elision for outer-letter deletion: `(t)E(a)`
   - Asterisks or "anagram of" for anagrams: `*(LOIN MEAT CLERIC)`
     or `Anagram [unusually] of TRIBAL YET`
   - "hidden in" / "buried in" + spanning words for hidden
   - "X reversed" / `X<` for reversal
   - "X containing Y" / "Y in X" for container
   - `+` between pieces for charade concatenation
2. Identify **leaf sources** (the words contributing letters) and
   their values.
3. Identify the **indicator** word for each operation (the bracketed
   `[unusually]` or the parenthetical or the inferred adjacent word).
4. Identify the **definition phrase** — the un-marked-up portion of
   the clue, or what the blog explicitly labels.
5. Identify **link words** — clue tokens not consumed by leaves /
   indicators / definition.
6. Build the Form per our schema and run through clipboard_verifier.

#### Reuse of existing prototype

The existing prototype implementation in
`prototypes/universal_form_v2/blog_parser.py` covers a meaningful
subset of blog patterns (see its top-level docstring for the list).
Plus `db_anchored_mapper.py` (which uses the DB as anchor for piece
identification) and `assembly_enumerator.py` (which composes Forms
from role-tagged pieces).

Before reusing, the prototype needs harmonising with current
discipline:

- Switch to `surface.tokenize` for clue tokenisation (currently uses
  `[A-Za-z]+` regex, which fragments "bird's").
- Ensure indicator words are propagated to the Form's nodes (the
  existing assembly tends to drop them when the production solver
  doesn't supply one).
- Drop dependency on the older `verifier.py`; check via
  `clipboard_verifier.py` only.
- Adapt to write to the seeder's storage layer instead of the
  shadow_pipeline JSONs.

#### Failure modes

Same two-tier classification as the JSON translator:
`translation_error` (parser couldn't recognise the blog structure)
and `verifier_fail` (Form built but didn't pass).

#### Source-specific dialect

Times TFTT blogs are typically terse (curly-brace + minimal prose).
15x15 Guardian/Independent blogs lean toward fuller prose. The
parser must handle both. Existing prototype handles TFTT well; 15x15
prose-heavy cases are weaker. Operational note: stratify early
batches by source to surface dialect-specific gaps.

-----

## 4. Verification pass

For each translated Form:

```python
verdict = clipboard_verifier.verify(form, clue_text, db, shadow_conn)
```

The verdict is recorded with all checks and enrichment candidates
(which are already structured per the verifier's output schema).

-----

## 5. Storage

### 5.1 PASSes

PASS forms write to the existing shadow_db tables:

- One row in `solves` carrying `clue_id`, `signature` (the canonical
  template id), `verdict='PASS'`, `answer`, `form_json`,
  `checks_json`, `enrichments_json` (empty), `created_at`
- One row per word/span in `solve_assignments`
- Any enrichment candidates (none expected on PASS) ignored

These are indistinguishable from solves produced by the runtime
cascade. The catalog regen treats them identically.

### 5.2 FAILs

A new shadow_db table is added for translation+verifier failures:

```sql
CREATE TABLE IF NOT EXISTS seed_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    source TEXT,
    puzzle_number TEXT,
    clue_number TEXT,
    direction TEXT,
    clue_text TEXT,
    answer TEXT,
    structured_explanation_id INTEGER,    -- the source SE row id
    components_json TEXT,                  -- the original components JSON
    translated_form_json TEXT,             -- the Form we built (if any)
    failure_kind TEXT,                     -- 'translation_error' | 'verifier_fail'
    failure_detail TEXT,                   -- error message or verifier's check details
    enrichments_json TEXT,                 -- if verifier_fail, candidate rows
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_seed_fail_clue ON seed_failures(clue_id);
CREATE INDEX IF NOT EXISTS idx_seed_fail_kind ON seed_failures(failure_kind);
```

This keeps FAILs human-reviewable. A simple report can list:

- count by failure_kind
- count by clue source
- top failing wordplay_types
- per-clue: clue text, answer, what we tried, what failed

The user said: "oftentimes a human eye can solve it quickly". The
report enables exactly that.

### 5.3 Enrichment candidates

When the verifier returns enrichment candidates (Rule 2 leaf failure
or indicator failure), they go into the **same shadow vocabulary
tables** the runtime cascade uses (`synonyms_pairs`, `indicators`,
etc.) with provenance. This is no different from the runtime path.

-----

## 6. Catalog regeneration

After a batch:

1. Run `extract_catalog.py` (extended to read from `shadow_db.solves`
   in addition to the existing JSON file inputs).
2. Catalog templates with ≥1 PASS form are present; new templates
   appear automatically; existing templates get new examples.
3. Re-run the verify-state checks to confirm no regression.

-----

## 7. Metrics / reporting

After each batch, emit a markdown report covering:

- **Counts**: candidates pulled, translated, verifier-PASS,
  verifier-FAIL, translation-error.
- **By model_version**: how each phase's solves translated.
- **Top new templates**: signatures previously not in catalog,
  with example counts.
- **Top failing wordplay_types**: which classes are hardest to
  translate or verify.
- **Coverage delta**: catalog template count before/after; form
  count before/after.

Reports go to `prototypes/universal_form_v2/runs/seed_report_<n>.md`.

-----

## 8. Component pieces

| Piece | Status | Notes |
|---|---|---|
| Source query | Trivial — one SELECT | Lives in the seed runner |
| Translator | **NEW** | `prototypes/universal_form_v2/seeder.py` (or similar) |
| Verifier | Reuse `clipboard_verifier.py` | No changes |
| Storage | Mostly reuse `shadow_db.py` | Add `seed_failures` table |
| Catalog regen | Extend `extract_catalog.py` | Read from solves table too |
| Report generator | **NEW** | Renders markdown per batch |
| Batch runner | **NEW** | `prototypes/universal_form_v2/runs/seed_batch.py` |

-----

## 9. Operational discipline

### 9.1 Per-batch flow — pre-filter, translate, save, eliminate

The seeding pass is **iterative** and exploits the existing matcher
as a duplicate-filter:

1. **Pre-filter with the matcher.** For every clue in the batch, run
   `tree_matcher.match_signature` against every template in the
   current catalog. Any clue that matches and verifier-PASSes is
   **already covered** by the catalog — translating it adds another
   example to an existing template, not a new template. Mark these
   as *covered* and **eliminate them from this batch's translation
   queue**. (Their solves go into shadow_db.solves so the per-clue
   record exists, but they don't drive catalog growth.)
2. **Translate the remainder.** Only clues that the matcher couldn't
   match against any existing template enter the translator. These
   are the candidates for genuinely-new templates.
3. **Verify.** Each translated Form passes through clipboard_verifier.
4. **Save.** PASSes go to `shadow_db.solves`/`solve_assignments`.
   FAILs go to `seed_failures` with full diagnostic detail.
5. **Eliminate from corpus.** Mark every clue processed in this
   batch — covered, PASS, or FAIL — as done so the next batch
   doesn't re-process them. (Translator improvements can later
   re-process FAILs explicitly; default is no-redo.)
6. **Re-run extract_catalog** so any new templates appear; the next
   pass's pre-filter is then richer.

The corpus naturally **shrinks** each iteration: more catalog
templates means more pre-filter coverage means fewer clues need
translation. Eventually the unprocessed remainder is either
translation errors, verifier fails (DB gaps), or genuinely novel
shapes worth user inspection.

### 9.2 Small batches

Per `feedback_small_batches.md`, do **not** run the full corpus on
the first attempt. The procedure is:

1. **Batch 1** — 20 candidates, mixed model_versions. Run, review
   the report carefully (every record), learn what the translator
   gets right and wrong.
2. **Batch 2** — 50 candidates, informed by batch 1's findings.
3. Subsequent batches scale up only as confidence accumulates.

The user reads the FAIL log between batches and decides whether to
proceed.

### 9.2 What "review" means

Between batches, the user (or Claude with the user) inspects:

- Failure samples — does the translator's interpretation match the
  intended solve?
- New templates — are the new signatures structurally honest, or are
  they artefacts of translator bugs?
- Enrichment candidates — are they real or spurious?

Adjustments either fix the translator or note a known limitation
before the next batch.

### 9.3 Stop conditions

Stop the seeding pass at any time and resume later. State is in the
shadow DB (idempotent re-runs by clue_id should be safe — the
schema's writes are insert-or-replace style at the (clue_id,
signature) granularity).

-----

## 10. Open questions / decisions deferred

- **Stratification by model_version**: do batches mix freely or
  group by source? Default: group by model_version for first
  batches; mix for later batches once each is well-understood.
- **Re-running with refined translator**: after a batch's failures
  inform a translator fix, do we re-run the same clues to see if
  they now PASS? Default: yes, idempotent re-run on previously-FAIL
  rows.
- **Stripping fortnight-old translation_errors**: when the translator
  improves, are old translation_errors cleared automatically?
  Default: keep the history with a `superseded` flag, never drop
  evidence.
- **Cross-feeding with the runtime cascade**: should the runtime
  cascade also write to shadow_db, and the seeding pass treat its
  output the same as production's structured_explanations?
  Tentatively yes — the same machinery.

-----

## 11. Decision log

- **2026-05-06** — design drafted. Source = structured_explanations
  (full table, not just claude_review). Translator uses components
  JSON. FAILs kept in dedicated `seed_failures` table for human
  review. Small-batches discipline applies.
- **2026-05-06** — per-batch flow refined: matcher acts as a
  duplicate-filter (clues matching existing templates are eliminated
  before translation); translator processes only the unmatched
  remainder; processed clues are marked done and removed from the
  corpus; corpus shrinks iteratively as the catalog grows.
- **2026-05-06** — second source added: raw blog text in
  `clues.explanation` (TFTT for Times, 15x15 for Guardian and
  Independent). Existing prototype `blog_parser.py` /
  `db_anchored_mapper.py` / `assembly_enumerator.py` reusable as the
  starting point but must harmonise with current discipline
  (`surface.tokenize`, indicator propagation, clipboard_verifier).
  Two translators feed the same verify-save-eliminate loop; duplicate
  Forms across sources deduplicate at the (clue_id, signature) level.
