# HANDOVER 2026-07-21 — container_deletion FALSE PASS + engine audit

Cold-start document. Every claim below is either VERIFIED (with the evidence/command that
proved it) or explicitly marked UNVERIFIED. No recommendation is given — the design decision
is left open for the user. Verify anything before acting on it.

---

## 0. GIT / STATE OF PLAY

- Branch `redesign`. HEAD = **ffbe79e5**. Ahead of origin/redesign by **2** (NOT pushed):
  - `1149abe3` mobile fixes: pattern-matcher spacebar + nav overlap
  - `ffbe79e5` pattern matcher: kill iOS double-space full stop (period normalisation)
- Uncommitted: `.claude/settings.local.json` (local config noise); **`scripts/worklist_probe.py`**
  (untracked — a read-only cascade probe built this session, see §6).
- No DB or engine code was changed this session. The only DB write this session was the user
  marking clue 10080621 status='invalid' (see §1).

---

## 1. THE FALSE PASS — clue 10080621 (Guardian 30064, 11a) BANKRUPTCY

Clue: "Left and right wings of party turn back, collapsing into ruin" (10) = BANKRUPTCY.
BANKRUPTCY is the correct answer (verified against `scraper/guardian/guardian_cryptic_30064.json`).
The stored parse was a false pass: it produced the right answer via wordplay that does no work.

Current status: **invalid** (the user marked it invalid mid-session). solved_by=container_deletion.

**Verified assembly** (by instrumenting `core.container_deletion_engine._build` and running the
real cascade read-only):
- op = `outer` deletion; reconstructed string I = `PBANKRUPTCYU` (p=1, L=10)
- inner = `BANKRUPTCY` from the single word "collapsing" (mechanism synonym)
- outer = `PU` from "party"+"turn" (party→P, turn→U)
- delete first (P) and last (U) of I → BANKRUPTCY
- `wfw_link` shows ALL 10 answer positions map to source_index=1 ("collapsing"); source 0
  ("party turn"=PU) contributes ZERO surviving answer letters — it is added then deleted.

**Where "collapsing"→BANKRUPTCY comes from** (verified): NOT a synonyms_pairs row.
`core.live_db.LiveDB.get_synonyms('collapse')` returns BANKRUPTCY via its bidirectional
definition-answer pass, sourced from `definition_answers_augmented`:
`definition='bankruptcy', answer='COLLAPSE', source='fifteensquared'`. Lemma: `_match_variants('collapsing')`
= ['collapsing','collaps','collapse']; the value attaches to the 'collapse' variant.
party→P and turn→U are real rows in `wordplay` (turn→U source 'admin'). "collapsing" is
independently typed as an anagram indicator in `indicators`.

---

## 2. DEFECT DEFINITION (as characterised this session)

Two candidate signatures of a "does-no-work" false pass:
- **(A) whole-answer piece**: a `wfw_piece` with role='source' whose value == the whole answer.
- **(B) net-zero piece**: a role='source' piece whose `ord` never appears in any `wfw_link.source_index`
  for that clue (contributes no surviving answer letter).

IMPORTANT EXEMPTION (verified via a false positive): a legitimate NAMED DELETION has a
removed-letters piece that is correctly net-zero. Clue 10075561 (AILMENT, handsolver):
"problem with train"=DERAILMENT minus "the German"=DER → AILMENT. The DER piece is net-zero
because it is deleted — this is CORRECT, not a defect. So signature (B) alone is not the test;
a removed-letters piece (mechanism deletion_removed) must be exempt.

`store.py:54` confirms role='source' `ord` == `source_index` (so the scan mapping is sound).

---

## 3. CORPUS AUDIT (read-only scan of all current passes — evidence)

Scanned every wfw_solve status='pass' row (~2184 at scan time).

Signature (A) is dominated by LEGITIMATE mechanisms where value==answer by design and is NOT a
defect: hidden (138), dd (104), homophone (19), alternation (3), plus manual (69) / prefill (4)
(human/AI committed). Suspect (A): catalog (2, legacy), container_deletion (1).

Signature (B), net-zero, filtered by inspection — 6 total:
- container_deletion (2): **10076624 UDDER**, **10076718 CAMPER** — both status=pass NOW (live).
- catalog (3, LEGACY engine): 1778035 LAIDUP, 1847196 LOIN, 9964853 TRIANGLE — status=pass NOW.
- handsolver (1): 10075561 AILMENT — **inspected: FALSE POSITIVE** (legit deletion, see §2).

**Confirmed live false passes (individually inspected):**
- 10076624 UDDER — "Milk container — shake after removing top": 'shake'→UDDER (whole-answer piece)
  + 'top'→T (net-zero). Real reading is a plain deletion (JUDDER minus its top letter). VERIFIED shape.
- 10076718 CAMPER — "Skips about naked in motor home": 'Skips'→SCAMPER carries all letters;
  'about'→C is net-zero. VERIFIED shape.
- 1778035 LAIDUP (legacy catalog) — inspected, same whole-answer+net-zero shape.
- 10080621 BANKRUPTCY — already status=invalid.

**NOT individually inspected** (flagged by scan only): 1847196 LOIN, 9964853 TRIANGLE (both legacy
catalog; 9964853 had an odd dead piece '14'→'' that may be a different artifact). Verify before acting.

**Every other reconstruction engine produced ZERO net-zero/whole-answer passes.**
CAVEAT: this measures MANIFESTATION in current passes, not CAPABILITY. A clean engine here is
NOT proven immune — it just has not hit a triggering clue. Signatures (A)/(B) are a floor, not
a ceiling (a piece could contribute one letter and still be spurious in a way the scan misses).

---

## 4. ENGINE ARCHITECTURE (verified facts)

- `core/container_deletion_engine.py` is ANSWER-DRIVEN (its own docstring): it reconstructs a
  pre-deletion string from the answer by ADDING letters, splits into outer/inner substrings, and
  checks each is makeable from clue-word values. It has NO guard against a piece equal to the
  whole answer, or against an added piece that the deletion removes in full.
- ~50 engines wired in `core/engine_registry.py`. 10 are catalog/signature-driven (the
  `*_signature_engine` modules + the deletion signature via `signature_verifier`). The rest
  reconstruct from the answer.
- The term "answer-driven" appears in nearly ALL engines (incl. the signature ones) and there it
  means "pieces verified to reconstruct exact answer letters" (anti-fabrication). So the label
  does NOT by itself isolate the defect.
- Each engine's `_verify` checks: letter coverage, unaccounted words, definition present,
  `role_validity.unbacked_roles`. It does NOT check that each operation does net work, nor that
  no single piece equals the whole answer.
- The AI is wired into the cascade engines as AUGMENTING CALLBACKS (`suggest_piece`,
  `define_fallback`) — e.g. `anagram_charade_engine.py:229` `augmented_lookup(lookup, suggest_piece)`.
  On an AI-on solve the same engines run and the AI supplies the piece values the DB lacks, folded
  into the same assembly. So the AI works THROUGH these engines; it is not a from-scratch
  re-derivation. VERIFIED for the suggest_piece/define_fallback path.
  UNVERIFIED: whether the separate headless nightly prefill agent (`claude -p nightly_prefill.md`,
  files via `core.prefill_commit.file_pending_prefill`) additionally reads the stored fail pieces.
- Forward-only: `store.persist` applies on (re-)solve; per the user, old puzzles are not re-run,
  so existing stored passes (correct AND incorrect) persist untouched until a clue is re-solved.
  Every solve (pass or fail) persists its pieces/links (near-miss assembly is stored on fails).

---

## 5. OPEN DECISION (NOT decided; no recommendation given)

How to handle the false-pass class was discussed but not decided. Options raised, stated
neutrally:
1. A soundness invariant that FAILS only unsound parses: "every source piece must contribute ≥1
   surviving answer letter, except a removed-letters (deletion) piece." Could live centrally in
   the registry finalisation (all engines) or scoped to container_deletion._verify. Would need a
   regression backtest (see §6) proving zero pass→non-pass on re-run; the AILMENT case is why the
   removed-letters exemption is essential.
2. Neuter container_deletion to always-FAIL.
3. Neuter to always-PENDING. NOTE (verified in registry): pending is terminal in the cascade — a
   pending result stops downstream engines from running.

Because changes are forward-only (§4), NONE of these fixes the false passes ALREADY stored as
passes. Those (10076624 UDDER, 10076718 CAMPER; legacy catalog 1778035/1847196/9964853) remain
status='pass' and are served until cleaned directly (as 10080621 was manually set to 'invalid').

---

## 6. TOOL: scripts/worklist_probe.py (read-only, untracked)

Built this session. Runs the LIVE cascade on a clue read-only — proven not to write
(nulls `wiring["store"]`, passes clue_id=None, never sets auto_signature; verified wfw_solve and
pending_enrichments row counts unchanged after a 150-clue run). Usage:
- `python -m scripts.worklist_probe <clue_id> [...]` — human-readable parse/status/warnings
- `python -m scripts.worklist_probe --run <ids>` — terse {status,engine,operation} for diffing
- `python -m scripts.worklist_probe --selfcheck N` — re-run N current passes
`run_set(ids)` returns {cid:(status,engine,operation)}; diff two runs (before/after a change) for a
zero-regression backtest. NOTE the selfcheck revealed wfw_solve's stored `status='pass'` is a STALE
baseline (645 legacy 'catalog' passes from 2026-06, older engine architecture) — a live regression
baseline must be freshly computed from the current cascade, not read from wfw_solve.

---

## 7. OTHER WORK THIS SESSION (not related to the engine issue)

- **Mobile template fixes — DEPLOYED + committed + browser-verified LIVE** (justcordelia.com):
  - `web/templates/tools_pattern.html`: pattern-matcher input now converts space AND period → '?'
    in the `input` handler (iOS forms the double-space full stop at the compose layer before the
    handler sees it), plus autocorrect/autocapitalize/spellcheck off. Commits 1149abe3 + ffbe79e5.
  - `web/templates/base.html`: nav given `flex-wrap` so the menu drops below the logo on phone
    widths instead of overlapping the "Cordelia" wordmark. Desktop unchanged (nav height 57px).
    Commit 1149abe3.
  - Both scp'd to the droplet and `systemctl restart cordelia`. The droplet is 2 template files
    ahead of the last full deploy.
- **Enrichment "home → IN" — NOT a bug** (Times 29600, SPAIN). The clue-page Accept button runs a
  LIVE query against the served DB (`web/routes/clue.py:534`). home→IN is in the LOCAL
  cryptic_new.db (source 'admin') but NOT on the prod droplet (verified via ssh). So on prod the
  Accept correctly shows — it is a pending DB deploy (local reference-DB additions not carried to
  the droplet), not a false enrichment.
- **Engine worklist review (replacement_letter [1]) — PARKED with a saved DECISION.** See
  `memory/engine_worth_only_if_valid_passes.md`: a new engine is only worth building if it
  produces confident valid PASSES; a pending-only engine that only turns FAIL→PENDING is the same
  human workload as now. replacement_letter dropped (its distinctive cases have an unclued/
  answer-derived new letter → can only pend; the passable cases are already substitution_engine).
- `memory/violation_log.md` updated (2026-07-20 entry).

---

## 8. WHAT IS NOT VERIFIED (do not treat as fact)

- Whether the headless nightly prefill agent reads stored fail pieces (§4).
- That legacy-catalog 1847196 / 9964853 are the same defect (scan-flagged, not inspected; §3).
- The blast radius / capability of the other ~40 reconstruction engines (only manifestation was
  measured; §3).
- The true (setter's intended) parse of 10080621 / UDDER / CAMPER beyond the plain-deletion
  reading noted — not rigorously derived.
