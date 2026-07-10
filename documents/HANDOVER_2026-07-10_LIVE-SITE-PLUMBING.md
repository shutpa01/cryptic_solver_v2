# HANDOVER 2026-07-10 — Next job: BUILD the live-site plumbing (design is SETTLED)

**START HERE.** The design phase is COMPLETE and user-approved. Do not re-open settled
decisions. Read `memory/live_site_wfw_plumbing.md` + `memory/publish_first_process.md` +
`memory/postpub_diagnosis_design.md` in full before any code.

## 1. STATE (all committed AND pushed — origin/redesign = 6b10a6f6)
- Branch `redesign`, HEAD 6b10a6f6, clean tree, PUSHED 2026-07-10 (user approved).
- GitHub master carries 9 commits (2026-05-15..27) NOT in redesign = the PRE-RESET WFW
  work ("Preserve..."), verdict DEAD HISTORY — never merge; redesign is the true line.
- Local live site (port 5000) = web/run_dev.py from THIS checkout (verified via process
  cmdline). Admin solver = core/wfw_web.py on 5099. Same data/clues_master.db.
- The publish-first process is LIVE and proven: user solved 31285+31286 in ~10 min each
  via /hs puzzle mode; first post-publish diagnosis ran (9 data-only, sigs 1672-1674
  filed pending, engine_worklist table + /worklist page live).

## 2. THE SETTLED DESIGN (user's decisions — build to THIS)
1. **ONE APP eventually**: web/ absorbs the WFW clue page + /hs behind is_admin; 5099
   retires last. Until the port, 5000 links across to 5099.
2. **No dual-read, no old-system serving**: the site launches with a GRADUALLY built
   stock (~a week of DT+Times+Guardian, all publish-first). Old
   clues/structured_explanations data RETAINED in DB, never served, never deleted.
3. **Hints**: 4-step ladder unchanged (tokened /reveal, hints.py:37-115). Step 1
   Definition / step 2 Wordplay type / step 4 Answer ← wfw parse fields. Step 3
   Explanation = a ONE-LINE SUMMARY built mechanically from the parse (assembly-line
   style: "RAG reversed + LAND → GARLAND").
4. **Full explanation = OVERLAY on the same page** (user's design: no page-flicking;
   "very easy way back to exactly where the user left"). Precedent: the tools overlay
   already in puzzle.html. Full WFW breakdown inside it.
5. **Admin flow**: puzzle list (admin) → click puzzle → /hs puzzle mode = the single
   solving surface. /hs gains a "Cryptic definition" button (sibling of &lit: no pieces,
   whole clue = definition, files PENDING). WFW clue page becomes VIEW-ONLY; its admin
   panel dies (everything else is already in /hs). "unforce" relocates (minor).
6. **Tier logic**: new puzzles are human-checked by construction → full hints; the
   confidence-tier machinery survives only for unserved legacy data.
7. **Old pipeline freeze rule**: a WFW-solved clue must NEVER be touched by the old
   /admin/rerun (admin.py:408-816 writes clues+structured_explanations — a parallel
   truth). Enforce before/while wiring.

## 3. BUILD PHASES (in order; each additive, tested, user sees it working)
1. **WFW read path** — new web-native reader (NO core imports; raw SQL on
   wfw_solve/wfw_piece/wfw_link) + the summary-line builder; wire hint steps to it.
   Walking skeleton: 31285/31286 render with WFW-fed hints on port 5000.
2. **Full-breakdown renderer + overlay** — user-facing WFW render (web styling, not the
   admin render) + "Show full explanation" overlay. Steal the PRINCIPLE of the old
   page-shape contract test (git show 671c5f6e:web/test_clue_wfw_render_contract.py —
   "every clue word visible, admin controls secondary") as a new regression test.
3. **Freeze rule + strip puzzle.html admin panels** (edit/rerun/approve/set-answer/
   DB+enrich/re-verify, all behind is_admin so users see no change) → replace with
   per-clue/per-puzzle links to 5099 (/hs?src&pnum + WFW clue page).
4. **CD button on /hs** + clue page → view-only. VERIFY FIRST: /setstatus captures
   signature_reviews (wfw_web.py ~:442) — /hsstatus must do the same before status
   moves wholly to /hs.
5. **Port WFW clue page + /hs into web/** behind is_admin (biggest chunk, last app work).
6. **Publish gate + daily automation** (3 papers/day: prefill → user walks → publish →
   post-publish diagnosis). 7. **SEO/crawler-visible render LAST** (indexing history).

## 4. RULES (unchanged, hard-won)
- Plain English, short sentences, bad news first. Verify before claiming (file:line).
- Questions are not instructions. All means all. One file at a time, test between.
- Engine changes A/B-gated (frozen-snapshot _regr.py pattern is in
  memory/publish_first_process.md — pending_store/catalog_loader/signature_queue path
  constants burned the first attempt).
- Claude never re-runs a clue for score; testing on temp DB copies is fine.
- Server restarts: tell the user to hard-refresh (Ctrl+F5); verify what the LIVE server
  serves before claiming visible. Env: .venv\Scripts\python.exe;
  $env:PYTHONPATH=(Get-Location).Path; solver 5099 (python -m core.wfw_web); site 5000
  (web\run_dev.py).
- Git: push only with explicit approval. Suggest checkpoint commits at milestones.

## 5. LOOSE ENDS (carry over)
- Engine worklist live at /worklist: replacement_letter (101 back-test candidates) is
  the agreed first engine build, then word_cycling (9). Deletion-family reveal audit
  (container_deletion etc.) still owed — same one-line fix + A/B as charade pair.
- Templates 1672-1674 pending; user promotes via sig-regress button when convenient.
- Daily flow continues during the build: prefill each new puzzle's fails into
  wfw_hs_assignments (rules in memory/publish_first_process.md — anagram-fodder not
  synonym-on-permuted-tiles; before/after/on = charade_positional indicators, not links).
