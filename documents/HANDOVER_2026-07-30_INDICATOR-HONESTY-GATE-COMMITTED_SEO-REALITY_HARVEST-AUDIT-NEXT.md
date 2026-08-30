# COLD HANDOVER — 2026-07-30

**One line:** Indicators now go through the review queue like every other enrichment (COMMITTED 0b0a59d0) — the silent-harvest hole is closed for NEW indicators. The Bing/SEO picture was verified end-to-end and is a waiting game, not a bug. Two open jobs: audit the ~986 already-harvested unreviewed indicators, and process the indicator review queue that will now start filling.

Read with the memory index (MEMORY.md). Everything below is labelled VERIFIED vs not.

---

## 0. BEHAVIOURAL NOTE FOR THE NEXT SESSION (read first)

The 2026-07-29 session was rocky and the user's trust is low, especially on SEO. Hold to these, hard:

- **Verify against the REAL running thing, then speak.** This session produced repeated guesses dressed as fact ("it's de-duped", "it's capped", "week-only deletion", "Bing saw them earlier") — each was wrong and the user demolished each in turn. Do NOT explain with a theory when the real artefact (the DB, the live page, the actual code path) is right there to check.
- **No grandstanding.** Long, clever-sounding answers read as bluff. Short, evidenced, plain.
- **The user often finds the answer faster than a broad investigation.** When they state a fact or a hypothesis, test THAT first, don't spelunk.
- **NEVER propose anything that changes the site's shape to search engines** (URL counts, sitemap size, removals/reinstatements). This is the confirmed cause of the Google collapse. Recovery = stability + time only.

---

## 1. MAIN WORK — indicator honesty gate (COMMITTED 0b0a59d0) ✅

**Problem the user found (VERIFIED):** indicators were the ONLY enrichment with no review path. On a prefill/AI reading, an indicator was stamped `source='manual'`, `confidence='high'` and harvested straight into the `indicators` reference table via `admin_db.add_indicator` — "silently approved", indistinguishable from a genuinely vetted entry, and then licensing future clues to pass. That is why `pending_enrichments` held ~0 indicators while the `indicators` table kept growing unreviewed. Synonyms/definitions already had the prefill→queue→review path; indicators had only the direct harvest.

**Fix (committed, mirrors the definition honesty gate exactly, NO engine touched):**
- `core/admin_db.py` — new `has_indicator(word, wordplay_type)` read-only check against the `indicators` table.
- `core/wfw_web.py` `_build_manual_parse` indicator branch — under `verify_db=True` (the prefill/AI reading): DB-backed → `source='db'`, harvested; reviewer-rejected → build refused; otherwise → `source='pending'`, `queue_indicator(...)` into `pending_enrichments`, and **kept OUT of the harvest**. Human `/hs` commit (`verify_db=False`) is UNCHANGED — the human is the authority and their indicator harvests as before.
- `core/wfw_web.py` `_confirm_prefill` — the "rests on an unbacked AI piece" refusal now also catches a pending **indicator** (indicators are annotations, which the old source-only scan missed), so Confirm/Approve-all refuses until the indicator is reviewed.
- The commit also LANDS the accompanying 2026-07-24 prefill **definition** honesty gate (`admin_db.is_definition` + the definition branch), because it shares the exact same diff hunks and is the pattern this mirrors. (It was verified-uncommitted; code is sound. The old open data-question — 20 already-CONFIRMED clues resting on unbacked defs — is unaffected by this commit.)

**Verification (VERIFIED through the real `_build_manual_parse`, the function `/prefillconfirm` calls; NOT through the live HTTP route — dev server was off, port 5001 contested):**
- Real clue 10073810, `verify_db=True`, `ok=True`: `'put on'` (in DB) → `source='db'`, harvested; `'with no top'` (not in DB) → `source='pending'`, NOT harvested.
- An unbacked indicator under `verify_db=True` wrote a row to `pending_enrichments` (0→1); the same under `verify_db=False` wrote nothing (human harvests). `has_indicator` checked against the live table. Both files + the staged versions compile. All test rows cleaned up.

**Operational change the user will now see:** a nightly-prefilled clue that needs an indicator NOT in the DB will now REFUSE to Confirm ("Confirm refused — ... indicator") until the user Accepts that indicator in the review queue. This is intended (parity with synonyms/defs) but it IS a workflow change — the review queue now carries indicators to action.

---

## 2. OPEN JOBS created/exposed by §1

1. **Audit the already-harvested indicators (soundness).** `indicators` table = 6,516 rows. By `source`: 5,117 NULL (original vetted corpus), **986 `'admin'`**, plus small named buckets. The 986 mix genuine admin-form additions with silently-harvested ones — `add_indicator` stamps EVERY caller `source='admin'`/`'high'`, so **provenance is lost and they cannot be told apart**. A wrong indicator harvested this way still licenses false passes. Decide: leave as-is, or build an audit (e.g. re-mark/spot-review recent high-id `'admin'` indicators). This is the "silently approved" set the user was worried about — the fix stops NEW ones, it does NOT clean these.
2. **The indicator review queue.** `pending_enrichments` currently holds **195** old `type='indicator'` rows (all `'anagram'`, mostly no puzzle_number — an old backfill). New prefill runs will now ADD real indicators here. The review/accept plumbing already works (`triage.apply_enrichment` handles `type='indicator'` → `add_indicator`). The user should start processing this queue.

DBs: `pending_enrichments` + `wfw_*` live in `data/clues_master.db`; the `indicators` reference table lives in `data/cryptic_new.db`. Both gitignored, backed up twice daily.

---

## 3. SEO / Bing — VERIFIED reality (no action, it's a waiting game)

Full detail in memory `bing_indexnow_list_vs_chart_and_index_reality`. Headlines, all read off Bing's live UI / the live site this session:
- **Submission works.** BWT IndexNow "Submitted URLs" chart = real count (428 on 27 Jul, 89–148 ordinary days). The "Submitted Urls list (latest 1000)" is a DE-DUPLICATED discovery view (~287 rows) that reads 1–3/day — MISLEADING; do NOT count its rows as daily volume. One submission path only (`scripts/indexnow_notify.py` via deploy), ~30 URLs/puzzle, ledger records only on all-200. The IndexNow per-puzzle ledger (2026-07-29, commit e972194c) is live and healthy (63 puzzles, one send each).
- **"Indexed ~1,459" (Site Explorer) is STALE** — it counts old pages that now 410. VERIFIED: clue 10061129 (Bing last-crawled 30 Apr, shows 200) returns 410 Gone live now; a recent clue is live+indexed. Not real coverage.
- **What Bing actually surfaces is ~nothing:** `site:justcordelia.com` returns no real pages (padded with foreign junk); whole-site ~962 impressions / 22 clicks, likely mostly the user. Serving set = 1,858 live clue pages across 25 dates (cumulative Jun→Jul; "keep everything, build daily" — there is NO week-only deletion; that earlier claim was wrong).
- **Real bottleneck = ranking/authority, not submission.** BWT: "not enough inbound links from high quality domains"; Site Explorer **Backlinks: 2**. Near-zero authority is why indexed pages don't surface. No technical lever fixes this. Backlinks come from participation (communities), not from ranking, so it is not a fully sealed loop — but it is a marketing/audience problem, the user's call, not code.

---

## 4. STILL-UNCOMMITTED at handover (HEAD = 0b0a59d0) — do NOT bundle blindly

`git status` tracked-modified:
- `core/admin_db.py` — the 2026-07-24 **mesh-promotion** fix in `add_synonym` (the honesty-gate `is_definition`/`has_indicator` parts are now COMMITTED; this mesh hunk was deliberately left out).
- `core/wfw_web.py` — the HS **CD auto-pass / INVALID-needs-comment** fix (2026-07-26) + the 07-26 **punctuation-not-a-link** rule + any earlier residue (the honesty-gate hunks are now COMMITTED; these were deliberately left out).
- `core/container_deletion_engine.py` — CD false-pass guard (07-22, NOT deployed).
- `scraper/danword/danword_lookup.py` — prize-toughie grid whole-number match fix (VERIFIED, 07-29).
- `scraper/orchestrator/daily_scraper.py` — HTML-tag strip at sync chokepoint (07-26).
- `web/run_dev.py`, `web/templates/about.html`, `web/templates/puzzles.html`, `web/wfw_read.py` — port 5001 / title SEO / OMELETTE fix (older).
- `.claude/settings.local.json`.

The two files I committed from (`admin_db.py`, `wfw_web.py`) STILL show modified because only the honesty-gate hunks were staged. When committing more, stage hunks — the handover-standing rule "don't blind-commit these files" still applies.

---

## 5. ENVIRONMENT / traps

- **Port 5001 contested** between this project (V2) and the separate AI_Solver project — both hardcode 5001, so a stale/wrong process can answer and make edits "appear to do nothing." After any dev restart, verify exactly ONE listener owns 5001 and it's the V2 venv (`Get-NetTCPConnection -LocalPort 5001`). No V2 dev server currently running.
- Reloader OFF — any `.py` change needs a FULL dev restart; launch with the ABSOLUTE venv path (`.venv\Scripts\python.exe`).
- The live site 410s any direct `curl` (Cloudflare/nginx allowlist) — a 403/410 from `curl` proves nothing; use the browser (real path through Cloudflare) to check live status.
- Sacred rule: NEVER modify a working stage engine for an edge case — build/add a helper. The §1 fix obeyed this (it's in the web commit/gate path, not an engine).

---

## 6. SUGGESTED NEXT STEPS (user's call)

1. Start processing the **indicator review queue** (now that prefill queues indicators and Confirm blocks on them).
2. Decide on the **986-indicator audit** (§2.1) — whether/how to review the silently-harvested set.
3. Decide the remaining **uncommitted fixes** (§4) — the danword grid fix is cleanly isolated; the rest share files and need hunk-level commits.
4. SEO: hold stability, watch the Bing indexed count and GSC curve over weeks. No URL/sitemap changes.
