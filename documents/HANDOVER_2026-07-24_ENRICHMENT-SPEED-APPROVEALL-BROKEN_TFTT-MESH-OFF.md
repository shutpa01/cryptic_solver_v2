# HANDOVER 2026-07-24 — enrichment speed + Approve-All (BROKEN), TFTT/mesh disabled

Cold-start document. Every claim marked **VERIFIED** (with the evidence) or **UNVERIFIED**.
Written after a session with repeated process failures (guessing before establishing facts,
committing before verifying the real user-facing surface). Trust nothing here without re-checking.

Branch **redesign**. Nothing deployed. Dev reloader is OFF (see §6) — a running dev server must
be manually restarted after any .py change.

---

## 0. GIT STATE

This session's commits on redesign (newest first):
- `a5a4b869` enrichment: drop duplicate `only` hidden field in Approve-all form
- `04300d6f` enrichment: show Approve-all & Confirm for ANY count (incl. 1)
- `2b6c5b79` enrichment: Approve-all & Confirm button (one click per clue)  ← **DOES NOT WORK, see §1**
- `56c56a7f` perf: seekable NOCASE index for clue-page synonym coverage checks
- `e51a07b5` disable api_mw_mesh synonyms in the solver lookup
- `31d60195` disable TFTT+haiku Times explainer
- `99e21293` prefill coverage: db_derives (bidirectional lookup)
- `e4ff93d7` archive dead code + drop 2GB recovery snapshots
- `cdd6d24e` enrichment dedup: cross-table coverage + drop substitutions table + Accept clears pending

Uncommitted working tree (do NOT assume these are all this session's):
- `web/wfw_read.py` (M) — OMELETTE reversal+deletion summary fix (§4). UNCOMMITTED.
- `core/container_deletion_engine.py` (M) — CD guard from a PRIOR session (07-22). UNCOMMITTED.
- `web/templates/{about,puzzles}.html` (M), `web/indexnow.py` (M) — prior session. UNCOMMITTED.
- `.claude/settings.local.json` (M) — local config.
- Also DB-note edit to clue 10081161 (§5) was written directly to clues_master.db (not git).

---

## 1. ⚠️ APPROVE-ALL BUTTON — BUILT BUT NOT VISIBLE (THE UNRESOLVED BUG)

**User asked for:** an "Approve All" button on the enrichment review that approves every
enrichment for a clue AND confirms the solve in ONE click (individual Approve/Reject kept for
selective use). Must work for ALL clues.

**What I did:** added the button to `core/wfw_web.py` `_enrichment_block` (commits 2b6c5b79 /
04300d6f / a5a4b869) + a `/approveall` route + extracted `_confirm_prefill` helper (shared by
`/prefillconfirm` and `/approveall`).

**VERIFIED:** rendering `_enrichment_block` in isolation for pending-prefill clues that HAVE a
queued row (10081253/10081255/10081256) produces the button HTML with `action="/approveall"`.
The `_confirm_prefill` refactor was tested non-mutating (refusal path still refuses + doesn't
freeze). The pending→add-form mapping was unit-tested.

**THE BUG (UNRESOLVED):** user restarted the dev server and STILL sees no button. So the button
is on the WRONG SURFACE. **There are TWO separate admin enrichment surfaces and I never
confirmed which one the user actually reviews on** — this is the root failure of the whole task:

1. `core/wfw_web.py` `_enrichment_block` — the **/solver mounted admin app** clue page. Renders
   "Enrichment needed" + per-row **Approve / Reject** forms (`/enrich`, `/reject`) + a prefill
   **Confirm** block + Hand-solver link. Reads the `pending_enrichments` TABLE via
   `admin_db.pending_for_clue`. **← I put the Approve-All button HERE.**
2. `web/routes/clue.py` → `web/templates/clue.html` (route `/clue/<slug>`, admin-gated
   `g.is_admin`) — the **public clue page** admin panel. Renders per-word buttons
   **"Accept {type}: {value}"** (clue.html:190-203, `w.accept_target`), posting to
   `/admin/accept-enrichment/<clue_id>/<word_index>`. Driven by `accept_by_index` computed
   per-piece from the PARSE (clue.py:508-568), NOT the pending_enrichments table.

**NEXT THREAD MUST DO FIRST, before writing any code:** get the EXACT URL (or screenshot) of the
page where the user sees enrichment-to-approve and no Approve-All button. That single fact
decides which surface owns the button. Do NOT reason about it from one clue — I did that
repeatedly and wasted hours.
- If the user is on surface #2 (clue.html per-word "Accept" buttons) — which the last evidence
  points toward — the Approve-All button is ENTIRELY ABSENT there; it must be built into
  clue.html / clue.py, batching the per-word `accept_target`s + a confirm. My wfw_web button is
  then on a surface the user never looks at.
- If the user is on surface #1 (wfw_web "Enrichment needed / Approve / Reject") — the button
  should already render; then debug WHY the /solver app didn't pick up the restart (is /solver a
  separately-served process? is it the same web/run_dev.py? is there template/bytecode caching?).

**Data note that confused the diagnosis (VERIFIED):** clue 10081249 is a pending-prefill whose
parse has a `source='pending'` piece (Exchange of shots→RALLY) but NO `pending_enrichments`
row. So surface #1's block renders EMPTY for it while a per-piece surface (#2) would still show
an Accept. This inconsistency (piece flagged pending but never queued) is real and PRE-EXISTING;
it is a strong hint the user reviews on surface #2. Not chased.

---

## 2. ENRICHMENT SPEED — INDEX FIX (VERIFIED, committed 56c56a7f)

**Real cause of slow enrichment processing:** coverage/Accept checks written as
`lower(column)=lower(?)` on `synonyms_pairs` (1.35M rows) force a FULL covering-index SCAN
(lowercasing the column defeats the index). **VERIFIED** ~272ms per synonym check on the MISS
path (the common case — an enrichment exists because the pair is NOT yet in the DB), fresh
connection. Rewrote the two clue.py synonym checks to `word=? COLLATE NOCASE AND synonym=?
COLLATE NOCASE` → seeks the existing NOCASE index → **3.6ms (~75x)**, identical results verified.

**NOT fixed (still full-scan if ever called, VERIFIED via EXPLAIN):** `admin_db.has_synonym` /
`has_substitution` use the same `lower(column)=` pattern. They currently have NO live callers
(db_derives replaced the gate), so they're moot NOW — but if re-used, they're 274ms each. The
definition check already uses a functional index (idx_daa_def_ans) and is fast. wordplay (1.5K
rows) / indicators (6K) scans are negligible.

**db_derives (my 07-23 change) is FAST (5ms), not the slowness** — it uses the indexed norm/
NOCASE columns. I initially suspected it and was wrong (measured).

**Caveat:** whether the index fix helps the surface the user is actually on depends on §1. The
fix targets clue.py (surface #2). If the user is on surface #1 (wfw_web), the slow query there is
`admin_db.pending_for_clue` (small pending table — fast) — so surface #1 was never the slow one.
This is UNVERIFIED against the real page.

---

## 3. APPROVE-ALL SPEED REASONING (design, not yet realisable — see §1)

The button batches N approvals into 1 HTTP round-trip + 1 render, but does NOT itself remove the
per-check cost; the index fix (§2) does. Ordering (index first, then button) was the plan. Moot
until the button is on the right surface.

---

## 4. OMELETTE SUMMARY FIX — reversal+deletion (VERIFIED, UNCOMMITTED web/wfw_read.py)

Clue 10081158 (TELEGRAPH 31300 17d, OMELETTE): the one-line summary showed "fruit→LEMON" and
silently dropped the removed N (LEMON reversed, less N → OMEL). Fixed `_describe` (web/wfw_read.py):
a piece whose placed letters aren't a forward sub-selection of its value but whose REVERSED
placed letters are, is a reversal-with-deletion → show "reversed less N". **VERIFIED** through
load_breakdown: summary now "fruit→LEMON reversed less N", regression cases (plain deletion /
plain reversal / no-op) unchanged. UNCOMMITTED. Full-breakdown view also correct. Separate small
gap noted (not fixed): the deletion indicator word "mainly" gets no labelled row.

---

## 5. LETTER-SHIFT TAG REMOVAL (VERIFIED, DB write to clues_master.db, not git)

Clue 10081161 (TELEGRAPH 31300 23d, MIDAS): user wanted the "move last letter to front"
(`last_front`) subtype removed so the indicator is a generic letter-shift usable for any shift.
Did a one-row UPDATE of `wfw_piece.note` from `letter_shift/last_front indicator` →
`letter_shift indicator`. **VERIFIED** via load_breakdown: renders "Letter shift" / generic
indicator, pass intact. **NOT done (flagged to user):** system-wide the hand-solver still FORCES a
last_front/first_end subtype when tagging letter_shift (core/admin_db.py:402 validation +
core/wfw_web.py:84-85,111-112 picker). If user wants generic letter_shift when TAGGING new clues,
loosen those.

---

## 6. ENVIRONMENT FACTS (VERIFIED)

- **Dev reloader OFF** (web/run_dev.py:14-21, use_reloader=False, dated 2026-07-12). Reason: the
  auto-reloader restarted the worker mid-render when the WFW clue page's lazy imports touched the
  watched tree, killing the request (clue page died ~15-25s; 13.5s with reloader off). debug=True
  keeps tracebacks + TEMPLATE auto-reload; CODE changes need a manual restart. **This is why code
  edits don't show until restart.** (Possible future: narrow the watch set with extra_files/
  exclude_patterns to restore code hot-reload — UNINVESTIGATED, would need to reproduce the hang.)
- Two admin clue surfaces exist (§1). Do not assume which is in use.

---

## 7. LEGACY AI DISABLED THIS SESSION (see memory legacy_ai_disabled_2026_07_23.md)

- **TFTT+haiku Times explainer OFF** (commit 31d60195 + Windows task "TFTT Retry" Disabled). It
  ran at 4am, Haiku-parsed the Times-for-the-Times blog and OVERWROTE clues.definition /
  ai_explanation with WRONG definition spans (Criminal→IRATE, guards→COMMIE; ~74 in backlog).
  VERIFIED it was the source of the wrong Times definitions via structured_explanations
  (model_version tftt+haiku) + logs/tftt_retry_2026-07-24.log. Serving is wfw-based, unaffected.
- **api_mw_mesh synonyms EXCLUDED from solver lookup** (commit e51a07b5, core/live_db.py
  get_synonyms both passes). ~796k uncurated Merriam-Webster "related words" (~60% of
  synonyms_pairs). ROWS KEPT (reversible: delete the `source<>'api_mw_mesh'` AND-clause). Trade-off
  (STATED): clues that passed only via a mesh-only synonym now fail. **UNVERIFIED how many real
  solves this costs** — no controlled before/after was run.
- **Guardian/Independent fifteensquared explainer STILL ACTIVE** — same Haiku-parses-a-blog
  mechanism as TFTT, same wrong-definition risk, on-demand only (no scheduled task). NOT disabled.
- **Old "Cryptic Solver Nightly Pipeline" task NOT disabled** — needs ADMIN elevation (I run
  non-elevated; Access denied). It writes to the OTHER project's cryptic.db, NOT V2's — vestigial,
  not corrupting V2. User to run elevated: `schtasks /Change /TN "Cryptic Solver Nightly Pipeline"
  /DISABLE`.

---

## 8. WHY LAST NIGHT (07-24) LOOKED BROKEN (VERIFIED)

Two independent things, NEITHER a code bug:
1. **Nightly prefill + diagnosis FAILED** — logs/nightly_2026-07-24.log:83-87 "Failed to
   authenticate: OAuth session expired and could not be refreshed" (exit 1). CAUSE: the claude CLI
   auto-updated 2.1.217→2.1.218 on 07-23 13:03 (.claude/.last-update-result.json), and the next
   unattended run couldn't use the old OAuth session. User re-logged in 05:27 (token now valid to
   Aug 22). VERIFIED auth works now (ran headless `claude -p` → OK). So no readings/enrichment were
   filed last night — that is why the review queue looked empty of prefills.
2. Cascade pass count 07-24 was low (5) but this is NOT proven to be a regression — comparing
   aggregate pass rates across DIFFERENT puzzles is invalid (I did this and was corrected). The
   fail population is dominated by anagram "no signature matched" which is synonym-INDEPENDENT
   (mesh removal cannot cause it). Whether mesh-off cost any real cascade passes is **UNVERIFIED**.
   Prevention for the auth issue: disable the claude CLI auto-updater and/or alert on nightly auth
   failure — NOT done.

---

## 9. PROCESS — READ THIS

This session repeatedly: (a) guessed a cause and stated it with unearned confidence, correcting
only when challenged; (b) re-ran things to reproduce output that was already stored; (c) committed
a feature (Approve-All) before establishing the one fact it depended on (which surface the user
reviews on). The user (rightly) lost confidence. The correct order for ANY reported problem here:
read the run's own stored output/log FIRST; establish the exact user-facing surface (get the URL)
BEFORE designing; do not attribute to recent changes without a controlled test; verify through the
REAL page before committing. The user values quality over speed absolutely — a verified small step
beats a fast wrong one.
