# Telegraph 31251 — clue-by-clue audit

## CONCLUSION (2026-05-29)
The system normally gets the MAJORITY of the pieces needed to parse a clue. Where
it is let down is at ASSEMBLY and VERIFICATION, not piece-finding. Of 12 clues
audited, the pieces were right in the large majority (1a, 5a, 10a, 12a, 18a, 20a,
25a-after-fix, 26a); what failed them was downstream — wrong assembly (nesting,
phrase-split, ignored container, first-match flat charade) and verification/scoring
(unnamed indicators, definition coverage + space-matching, multi-word anagram
fodder, link words, polluted synonyms). The genuinely-wrong few (8d, 11a, 13a) trace
to bad inputs (fabricating Sonnet, truncated/junk synonyms, definition reuse); the
non-solves (15a, 23a) to a missing synonym and a hard compound.
Implication: fixing only assembly + verification (without touching the signature
engine) would massively raise the score, and several fixes are simple and broad —
name the indicator that produced a piece; normalise spaces in the definition check;
combine multi-word anagram fodder; treat link words as links; clean polluted
synonyms. The key pieces are in place; the work is real but well-targeted.



Cold run of 31251 through the full cascade (created 2026-05-28/29). Going through
clue by clue to see what each part of the engine actually did. Read-only; no DB
changes.

## Where things are stored (reference)
- Explanation text is dual-written: clues.ai_explanation AND
  structured_explanations.components.sig_explanation (identical copies).
- clues.explanation = the human/blog column (separate; often empty on cold runs).
- Score = structured_explanations.confidence; the display tier is derived from it
  (HIGH >= 0.70, MEDIUM 0.40-0.69, LOW < 0.40) by web/models.py compute_hint_tier.
- model_version = LAST writer only. NOT reliable for "who solved it" — later phases
  overwrite earlier solves (see 5a).

## Cross-cutting findings so far
- Sonnet still active: tier2_solver.py and solver.py both use claude-sonnet-4-6;
  results tagged haiku_sonnet_tiered_v1. Paid Phase 2 was NOT switched off.
- Failed parses (confidence 0.0) are written to the DB as rows anyway.
- model_version attribution is corrupted by later-phase overwrites; a correct
  earlier solve can end up tagged as a later engine's failure.
- PROMPT TRUNCATION (ROOT CAUSE OF MANY FAILURES): the Tier 2 Sonnet prompt
  (tier2_solver.build_prompt) instructs "capture the FULL multi-word phrase" ONLY
  for indicators (line ~119), NOT for synonym/abbreviation pieces. And the
  word-analysis it is constrained to (build_word_analysis) offers only single-word
  synonym roles plus 2-word adjacent phrase roles (wp_words[i:i+2]) — nothing 3+
  words. Result: multi-word synonym SOURCES collapse to a single word (e.g.
  "provocative posters"->TROLLS becomes "posters"->ROLLS with the spare T parked on
  "in"). These truncated pieces are then collected as enrichment gaps and queued.
  FIX DIRECTION: extend the full-multi-word-phrase instruction to synonym/
  abbreviation pieces, and offer longer phrase roles in the analysis. (Definition
  truncation is a separate prompt, haiku_definition — to be checked.)
- RE-VERIFY != RE-SOLVE: the reverify route only re-scores the EXISTING stored
  explanation; it never re-solves. So changing an enrichment (synonym/definition)
  and re-verifying does NOT update the explanation — the clue must be RE-RUN through
  the complete solver for the change to take effect.
- SYNONYMS_PAIRS CONTAMINATION (data-quality, causes false parses): synonym sets in
  cryptic_new.db synonyms_pairs are heavily polluted. Example: "first" has 187
  "synonyms" including obvious junk — archer, archest, boxcar, caught, enceinte,
  expectant (both = pregnant), boss, charge, du jour, banner. "first"->"big" sits in
  this set, so the solver used it and the verifier "VERIFIED" it. Junk synonyms
  produce confidently-wrong pieces AND pass synonym verification. This undermines
  synonym-based solving and verification generally. NEEDS a synonym-table cleanup.

## Clue log

### 8d DISPLAYS (conf 0.0, model haiku_sonnet_tiered_v1) — WRONG, paid
Clue: "Shows what negative theatre critics do" (8). Def: "Shows".
Stored parse: DIS (negative) + PLAY (theatre) + S (last letter of "critics").
- Came from a paid Sonnet 4.6 call (Phase 2), not the mechanical engines.
- The "S = last letter of critics" is fabricated — no last-letter indicator in the
  clue. Verifier correctly scored it 0.0.
- Correct parse is simpler: DIS (negative) + PLAYS (theatre = plays).
- Verdict: paid call produced a sub-mechanical wrong parse, scored 0.0, still
  stored.

### 1a DEMOCRAT (conf 0.2, model signature_solver_enriched_v1) — RIGHT PIECES, WRONG ASSEMBLY
Clue: "March followed by Queen perhaps hosting Republican politician" (8).
Def: "politician".
Stored parse: DEMO ("March") + CAT ("Queen perhaps") + R ("Republican")
[container: "hosting"]; assembly recorded as container inner=DEMO outer=CAT.
- Pieces and indicator all correct. True parse: DEMO + (CAT hosting R -> C-R-AT =
  CRAT) = DEMOCRAT — a charade whose 2nd part is a container.
- It nested wrongly (DEMO inside CAT) and left R dangling, so the reconstruction
  failed -> 0.2.
- Verdict: compound-clue weakness (container nested in a charade). Right
  ingredients, wrong nesting.

### 5a LAPSED (conf 0.0, model signature_solver_enriched_v1) — CORRECT SOLVE, CLOBBERED
Clue: "Returned some swedes, palpably out of date" (6). Def: "out of date".
Stored explanation (correct): hidden reversed in "swe DESPAL pably" [hidden:
"some"] -> DESPAL -> LAPSED.
- The mechanical HIDDEN engine (Phase 0) solves this — confirmed by running
  backfill_dd_hidden.try_hidden directly (returns reverse hidden in "swedes,
  palpably"). User correctly predicted this.
- model_version reads signature_solver_enriched_v1 at 0.0 — the LAST writer. A
  later Phase 3 signature re-run overwrote the original (almost certainly a Phase 0
  mechanical_hidden solve at 1.0) and clobbered the confidence to 0.0.
- Explanation, definition and wordplay_type stored are ALL correct; only the
  confidence (0.0) is wrong.
- Verdict: a correct, high-confidence hidden solve was overwritten and stored as a
  0.0 failure.

#### 5a — where the explanation is written, and why it's 0.0 (traced)
- Text built by sig_explain() in sonnet_pipeline/sig_adapter.py (is_hidden branch).
- Written by store_signature_result() (same file): writes to clues.ai_explanation
  UNCONDITIONALLY (no COALESCE) and to structured_explanations.components, tagged
  signature_solver_enriched_v1. The unconditional ai_explanation write is what
  overwrote the earlier Phase 0 hidden solve.
- The 0.0 is set inside store_signature_result: it re-scores with the strict
  ExplanationVerifier (sig_adapter.py ~line 803) and keeps the stricter score.
  Ran that verify directly — verdict FAIL, score 0:
    - definition 'out of date'->LAPSED: VERIFIED
    - hidden span reverses to LAPSED: VERIFIED
    - indicator: unverifiable — explanation does not name the reversal indicator
    - word_coverage: WRONG — 1/7 unaccounted: "returned"
- ROOT CAUSE (real bug): sig_explain's hidden-reversed rendering names only the
  HIDDEN indicator ("some") and DROPS the REVERSAL indicator ("Returned"). The
  verifier then correctly flags "Returned" as an unaccounted clue word and fails
  the whole (otherwise-correct) solve to 0. So a correct solve is stored at 0.0
  purely because the explanation builder omitted one indicator attribution.
- Correction to earlier note: the 0.0 is the strict ExplanationVerifier's verdict
  (inside store_signature_result), NOT confidence.py.

### 10a TASMANIAN DEVILS (conf 0.0, model signature_solver_v1) — RIGHT BITS, POOR ASSEMBLY/EXPLANATION
Clue: "Marsupials, wild native animals circling daughter and son" (9,6). Def:
"Marsupials,".
Stored parse: anagram of NATIVE [anagram: "wild"] + ANIMALS + D [anagram:
"circling"] + S = TASMANIANDEVILS.
- The bits are correct: NATIVE + ANIMALS + D + S genuinely anagrams to
  TASMANIANDEVILS; "wild" is the anagram indicator; daughter=D, son=S.
- WHERE THE BITS WERE DETECTED: the word-analysis stage of the SIGNATURE solver —
  word_analyzer.py (analyze_words/analyze_phrases), Phase 1. Per word it queries
  RefDB: get_abbreviations (daughter->D, son->S => ABR_F), anagram fodder (raw
  letters of native/animals => ANA_F), indicators (wild => anagram). Detection was
  good.
- POOR part is downstream: the explanation treats BOTH "wild" and "circling" as
  anagram indicators (double-attribution; "circling" should not be an anagram
  indicator), and the assembly is a flat "anagram of [everything]". That's in the
  matcher + sig_explain, not detection.
- NOTE (duplication): "word analysis" is implemented TWICE — Phase 0.5's
  batch_v1_solver per-word value gathering, and the signature solver's
  word_analyzer (Phase 1/3) — same RefDB, two separate codebases.
- TODO: check whether RefDB actually lists "circling" as an anagram indicator
  (root of the mis-attribution).

### 11a STROLLS (conf 0.0, model signature_solver_enriched_v1) — WRONG ("lucky letters")
Clue: "Ambles in front of sickeningly provocative posters" (7). Def: "Ambles".
Stored parse: S (first letter "sickeningly", ind "front") + T ("in") + ROLLS
("posters") = STROLLS.
- Verifier (ran directly) FAIL/0: definition VERIFIED; "in"=T VERIFIED (it IS in
  the DB); S first-letter VERIFIED; assembly S+T+ROLLS=STROLLS MATCH; BUT
  "posters"=ROLLS not in DB (unverified, likely enrichment-injected on the enriched
  pass); word_coverage WRONG — "of" and "provocative" unaccounted.
- Failure mode: "lucky letters" — pieces concatenate to the answer via a fabricated
  synonym while ignoring two clue words ("provocative" must feature in the true
  parse). Genuinely wrong; verifier correctly scored 0.0.
- Contrast: 10a = right bits/poor wording; 5a = right solve/missing one indicator;
  11a = actually incorrect.

#### 11a — enrichment provenance + re-solve learning (traced)
- The true parse is S (first letter of "sickeningly", "in front of") + TROLLS
  ("provocative posters" = internet trolls) = STROLLS. Def "Ambles".
- The bad enrichment posters->ROLLS: checked all tables.
  - NOT in reference DB (synonyms_pairs / def_answers) — no leak.
  - NOT in rejected_enrichments — no rejection record for this exact pair.
  - IS in pending_enrichments (queued from this 31251 run).
  - Related ROLL pairs ARE rejected: 'bakery'->ROLL, 'products'->ROLL (2026-05-13).
  - Rejections are keyed on EXACT (word, letters), so rejecting ROLL under other
    words does not block 'posters'->ROLLS.
- SOURCE of posters->ROLLS: the Tier 2 Sonnet prompt (see prompt-truncation finding
  above). Sonnet's piece decomposition truncated "provocative posters"->TROLLS to
  "posters"->ROLLS; that piece was collected as a gap and injected/queued.
- USER FIX: changed enrichment to "provocative posters"->TROLLS, but it was stored
  with a TYPO ("provocotive posters"), which does not match the clue's
  "provocative" — so the lookup would miss it until corrected.
- After correcting the typo, the clue PARSES correctly (S + TROLLS) but STILL SCORES
  LOW — separate scoring issue to investigate.
- LEARNING: re-verify only re-scores; to see an enrichment change take effect the
  COMPLETE solver must be re-run on the clue.

### 12a ENTHRAL (conf 0.0, model signature_solver_v1) — CORRECT WORDPLAY, DEFINITION COVERAGE MISS
Clue: "Have Spellbound rental playing for husband to get into" (7). Def: (none stored).
Stored parse: anagram of RENTAL + H [anagram: "playing"] [anagram: "into"] = ENTHRAL.
- Knew it was an anagram: "playing" is in indicators table as anagram, confidence
  very_high -> word_analyzer flagged it; letters RENTAL+H verified to anagram to
  ENTHRAL.
- husband -> H: DID get it. DB wordplay table has ('husband','H','single_letter').
  It's in the pieces (clue_word husband, letters H) but the explanation renders it
  bare ("+ H") with no "H (husband)" attribution — detection fine, rendering hides it.
- Definition: genuinely NOT found (None). COVERAGE GAP: ENTHRAL has many defs in
  def_answers_augmented (incl. "Hold spellbound"->ENTHRAL, grip, charm, captivate)
  but NOT the clue's exact "Have Spellbound" / bare "Spellbound". Edge-anchored
  DB-exact finder tried Have / Have Spellbound / into / get into -> nothing; Haiku
  fallback didn't rescue.
- Score 0.0 cascade: no definition -> verifier no_definition AND word_coverage
  WRONG (have, spellbound, for, husband, to, get unaccounted). Correct wordplay
  scored 0.0 purely because the definition phrase wasn't in the DB.
- Also: "into" wrongly tagged as a 2nd anagram indicator (same double-attribution
  bug as 10a).
- HAIKU DID KNOW (tested live): haiku_definition.find_definition returned None, but
  the raw Haiku reply was "Spellbound" — a correct synonym. Two failures killed it:
  (1) TRUNCATION — Haiku returned the single word "Spellbound" not the clue's
  two-word definition "Have Spellbound" (same single-word truncation as the synonym
  pieces, now confirmed in the haiku_definition prompt); (2) ORPHAN RULE — the
  validator requires the def edge-anchored with no real word orphaned; "Spellbound"
  matches at position 2, which orphans "Have" at the start, so it was discarded ->
  None. Had Haiku returned "Have Spellbound" it would have validated and 12a would
  likely have scored well (wordplay is correct).
- FIX: push Haiku (definition prompt) to return the FULL multi-word phrase, not one
  word; revisit the orphan rule's strictness.

### 13a HONEY BEE (conf 0.0, model signature_solver_v1) — GENUINELY WRONG (definition reused as wordplay)
Clue: "Darling Auntie brushing off second black insect" (5,3). Stored def: "black
insect". Parse: HONEY ("Darling") + BEE ("insect").
- Verifier FAIL/0: Darling=HONEY VERIFIED; insect=BEE VERIFIED; HONEY+BEE=HONEYBEE
  MATCH; but def 'black insect'->HONEYBEE not in DB; "off" is an anagram/deletion
  indicator the parse hides; word_coverage WRONG — auntie, brushing, off, second
  unaccounted.
- DEGENERATE/LAZY PARSE: it reused "insect" (part of the stored definition "black
  insect") as the BEE synonym, and ignored the real wordplay "Auntie brushing off
  second" plus the "off" indicator. This is the "definition reused as wordplay"
  failure (FIX-001) + ignored indicator.
- Genuinely wrong (like 11a), correctly scored 0.0. Engine couldn't crack the true
  BEE wordplay so it fell back to double-using the definition word.

### 15a RUDDY (has_solution=0, no SE row) — HONEST NON-SOLVE
Clue: "Largely bad-tempered, yelled on vacation, turning red" (5). Def "red" (stored).
- Completely unsolved: no structured_explanations row, no explanation, wordplay_type
  None. Only the definition "red" was stored (Phase 1 stores the definition even on
  solve failure).
- True parse is a triple/quad compound: RUD ("bad-tempered"=RUDE, "largely"=drop
  last) + DY ("yelled" outer letters via "vacation"=YD, "turning"=reversed->DY);
  RUD+DY=RUDDY, def "red".
- Building blocks checked in DB:
  - "Red"->RUDDY: PRESENT (def stored).
  - "largely": indicator parts/last_delete (truncation) PRESENT.
  - "vacation": indicator parts/outer_use (outer letters) PRESENT.
  - "turning": indicator reversal/general PRESENT.
  - "bad-tempered"->RUDE: MISSING. DB has bad-tempered->CROSS/mean/ratty/scratchy/
    stroppy but NOT RUDE. <-- COVERAGE GAP that blocks RUD.
- TWO failures at once: (1) coverage gap (bad-tempered->RUDE absent) blocks the RUD
  piece; (2) even with RUDE, the parse stacks synonym+truncation+outer+reversal+
  charade — beyond the single-pattern catalog.
- POSITIVE: the engine wrote NOTHING rather than a garbage 0.0 row. Honest failure.
  Contrast with 8d/11a/13a which fabricated nonsense parses. Note all 3 indicators
  and the definition WERE correctly available — only one synonym was missing.

### 18a EPSOM (conf 0.25 stored / MEDIUM on re-verify, model mechanical_v1) — CORRECT, docked by missing indicator attribution
Clue: "Racecourse records form regularly" (5). Parse: EPS ("records") + OM (even
letters of "form") = EPSOM, def "Racecourse".
- Verifier MEDIUM/50: definition VERIFIED; records=EPS VERIFIED; EPS+OM=EPSOM MATCH;
  "OM" as even of "form" positionally VERIFIED; word_coverage WRONG — 1/4
  unaccounted: "regularly".
- The explanation is correct, INCLUDING the even-letter extraction. The single
  failure: the operation that produced OM (even/alternate letters, signalled by
  "regularly") is described in prose ("even letters of form") but the indicator word
  "regularly" is NOT named, so it's flagged unaccounted; and wordplay_type stays
  "charade" with no "alternation" type recorded.
- SAME ROOT as 5a: explanation builder states a positional/operation mechanism in
  prose but doesn't attach the indicator word in verifier-recognised bracket form
  (e.g. [alternating: "regularly"]). FIX: name the operation's indicator word.
- CONFIRMED FOUND-BUT-NOT-NAMED: "regularly" IS in indicators (parts/alternate,
  alternate, alternating). The V1 charade solver only tries even/odd extraction when
  it sees such a parts indicator — so "regularly" is exactly what LICENSED the OM
  even-letters extraction. The engine detected and USED it; the explanation builder
  just didn't name it. Pure rendering/attribution bug, NOT a detection/coverage gap.
  Naming it alone would lift 18a from MEDIUM to a clean solve.

### 20a NAPOLEON (conf 0.0, model signature_solver_v1) — RIGHT BITS, KNOWN ANSWER, BOTCHED ASSEMBLY
Clue: "Sleep with ring on, embracing the French emperor" (8). Def "emperor".
True parse: NAP (Sleep) + O (ring) + ON, embracing LE ("the french") -> NAPO[LE]ON.
Stored parse (charade): NAP + O ("ring") + ON (from clue) + LE ("the") + LE
("French") -> claims 5 pieces = NAPOONLELE (10 letters != 8). conf 0.0.
- Bit sources (checked DB):
  - "ring"->O: real (wordplay, misc).
  - LE: real but as the PHRASE "the french"->LE (foreign_french; also LA/LES/L).
    The explanation MIS-ATTRIBUTES this single phrase as "the"->LE + "French"->LE
    (two pieces) and COUNTS LE TWICE. Neither single-word entry exists: "the"->T/TH,
    "French"->FR.
  - "embracing": IS a container indicator in DB (container/outside) — but UNUSED;
    op tagged flat "charade".
- ASSEMBLY FAILURE despite known answer: (1) duplicated LE (phrase split into two),
  (2) ignored the container indicator and did a flat charade -> 10 letters, no match.
  With pieces {NAP,O,ON,LE} and known answer NAPOLEON, the container insertion
  NAPO[LE]ON is trivial, but it was never attempted.
- Theme: phrase-synonym split/double-counted + container indicator ignored +
  first-match flat-charade. The hardest part (finding the bits) succeeded; the
  easiest part (placing LE given the known answer) failed.

### 23a CHIANTI (has_solution=0, no SE row) — HONEST NON-SOLVE + corrupted clue char
Clue (as stored): "What noisy soccer fans may do around one [REPLACEMENT CHAR] start
to imbibe drink" (7). Def "drink" (stored).
- Unsolved: no SE row, no explanation. Only definition "drink" stored. No nonsense
  written (good, like RUDDY).
- True parse: CHANT ("what noisy soccer fans may do") around I ("one") -> CHIANT, +
  I ("start to imbibe" = first letter) -> CHIANTI. Container nested in a charade
  with a first-letter extraction — compound beyond the single-pattern catalog.
- DATA DEFECT: the stored clue_text contains a corrupted character (a "replacement
  char" where a dash should be: "around one ? start to imbibe"). A scraping/encoding
  defect, separate from the solver, that can also disrupt tokenisation.

### 25a BIG DEAL (conf 0.0, model signature_solver_enriched_v1) — WRONG PARSE FROM POLLUTED SYNONYM
Clue: "I'm not impressed at first by Italian gelato portion" (3,4). Def "I'm not
impressed". Stored parse: BIG ("synonym of first") + DEAL ("synonym of portion").
- True parse: BIG = first letters (acrostic) of "By Italian Gelato" ("at first" =
  indicator) + DEAL ("portion"). Def "I'm not impressed" (= "big deal").
- Verifier FAIL/0: def not in DB; first=BIG VERIFIED; portion=DEAL VERIFIED;
  BIG+DEAL MATCH; "impressed" container indicator hidden; word_coverage WRONG —
  6/10 unaccounted: i, m, at, by, italian, gelato.
- ROOT CAUSE (corrected): NOT mere role-ambiguity. "first"->"big" is POLLUTED DATA —
  it sits in a contaminated 187-entry synonym set for "first" (archer, boxcar,
  enceinte, expectant, ...). The solver grabbed the junk synonym first->BIG, built a
  letter-matching but wrong parse, and the verifier "VERIFIED" the junk synonym. The
  acrostic of "by Italian gelato" (the real source of BIG) was ignored -> 6 words
  unaccounted -> 0.0.
- Lesson: polluted synonyms_pairs directly cause confidently-wrong parses that pass
  synonym verification. (See cross-cutting "SYNONYMS_PAIRS CONTAMINATION".)
- AFTER USER DELETED first->big AND RE-SOLVED (now model mechanical_v1, updated
  2026-05-29 15:10): the PARSE IS NOW CORRECT — B (first of "by") + I ("Italian") +
  G (first of "gelato") + DEAL ("portion") = BIG DEAL. Removing the junk synonym
  forced the real acrostic-of-initials decomposition. Strong proof that cleaning
  synonyms_pairs fixes parses.
- BUT still conf 0.0. Verifier on the corrected parse: def 'I'm not impressed'->
  BIGDEAL not in DB; portion=DEAL, B/G first-letters, assembly all VERIFIED;
  word_coverage WRONG 4/10: i, m, at, first. Residual 0.0 = THREE peripheral issues,
  none in the wordplay: (1) definition coverage gap; (2) "at first" first-letter
  indicator not NAMED -> at/first unaccounted (same as 18a/5a); (3) "I'm" apostrophe
  split into i+m, not absorbed by the definition match (tokeniser apostrophe quirk).
- LESSON: cleaning data is necessary but NOT sufficient — a now-correct parse still
  scores 0.0 due to definition coverage + unnamed indicator + apostrophe tokenisation.

### 26a HIT THE HEADLINES (conf 0.0, model signature_solver_enriched_v1) — CORRECT PARSE, VERIFIER BUGS
Clue: "Drunk inhaled his teeth and got media attention!" (3,3,9). Parse: anagram of
INHALED [anagram: "Drunk"] + HIS + TEETH = HITTHEHEADLINES; def "got media attention".
- Parse is COMPLETELY CORRECT: INHALED+HIS+TEETH anagrams exactly to
  HITTHEHEADLINES; "Drunk" named as indicator; "got media attention" is the def.
- Scored 0.0 purely from THREE VERIFIER BUGS (not the solver, not the data):
  1. MULTI-WORD ANAGRAM FODDER NOT COMBINED: verifier checked only the first fodder
     word ("'INHALED' anagrams to HITTHEHEADLINES: NO") instead of INHALED+HIS+TEETH.
  2. DEFINITION SPACE-NORMALISATION: the DB DOES contain "got media attention"->
     "HIT THE HEADLINES" (and "!" variant), but the verifier compared to space-
     stripped "HITTHEHEADLINES" and reported "not in DB".
  3. LINK WORD "and": flagged as unaccounted by word_coverage (should be a link).
- NEW CATEGORY: the VERIFIER itself failing a correct solve (vs earlier false-lows
  which were explanation-rendering or data issues). Per standing rule, verifier NOT
  touched — documented only.
- (Stored conf 0.25 vs re-verify 50 — minor scoring discrepancy from the
  mechanical_v1 path; not the main point.)
