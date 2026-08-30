# HANDOVER 2026-07-17 — SITE IS LIVE; NEXT = REVIEW TODAY'S PUZZLES

Cold-start document. Read this first, then the memory index (MEMORY.md). Plain
English; verify against the code before acting. **justcordelia.com went LIVE
2026-07-16** and is secured. The daily operating rhythm has now begun.

---

## 0. ★ THE NEXT SUBJECT — review today's (2026-07-17) puzzles

This is now the **daily job**, not a one-off. Each morning the night's scrape +
cascade + prefill have run; the job is to get today's serving-paper puzzles
(telegraph / times / guardian) fully solved and published. A puzzle only shows
publicly when **every** clue is served (the puzzle-level display rule), so the
review's goal is: no leftover clues.

**Read first (the operative process, in this order):**
1. memory/publish_first_process.md — THE settled flow (user's own design).
2. memory/feedback_leftover_process.md — quality standards for solving leftovers
   (the definition rule, coverage check, blog compass, honesty-over-score). NB:
   its `structured_explanations` round-trip is the OLD mechanism — the CURRENT
   mechanism is the /hs work-list + Commit (memory streamline_hs_replaces_triage:
   `/hs?src=X&pnum=N`; /triage retired). Take the STANDARDS from it, not the
   round-trip.
3. memory/nightly_triage_process.md + phase6_nightly_automation.md — what the
   night already did, and the diagnose-only role boundary.

**The four-step rhythm (publish-first):**
1. **Night (already ran):** scrape → nightly_cascade.py → prefill writes best
   readings into the /hs grid (wfw_hs_assignments) for FAIL/PENDING clues, only
   where no user state exists.
2. **Morning = THE REVIEW (this job):** the USER walks the puzzle in the /hs
   work-list strip, corrects any wrong reading, and hits **Commit (manual)** per
   clue — one decision: "is this reading right?" Commit's guards (all words
   accounted, all tiles covered, definition required) are the safety net;
   reusable pieces flow into the reference DB on every commit. Claude's role is
   to HELP solve/verify the hard leftovers to the quality bar — never to commit.
3. **Publish** — fast; nothing blocks on signatures/engines.
4. **Post-publish (non-critical time):** Claude DIAGNOSES the frozen manual
   solves (data / signature / engine gap) and files only the safe fixes
   (pending-only signatures, engine_worklist). See §"how diagnosis works" below.

**Tools:**
- The /hs work-list: /hs?src=<source>&pnum=<number> (source values: telegraph,
  times, guardian, dailymail, independent, cordelia — get these exact or you get
  0 rows).
- "Cascade now" button on the puzzle page (/admin/cascade/<source>/<pnum>) — the
  prize-morning path: solve grid → save answers → Cascade now → /hs work list.
- python -m core.diagnose <clue_id> — READ-ONLY. Two views: "pieces" (is the raw
  material in the DB, and does it reach the answer?) and "engines" (which engine
  got closest, what it left unaccounted). USE IT FIRST on any fail.

**Non-negotiable quality rules (from feedback_leftover_process.md — these ARE the moat):**
- Every clue MUST end with a definition = the phrase FROM THE CLUE. No skip option.
- Coverage check BEFORE any DB write: every synonym / abbreviation / indicator /
  definition claim must already be in the DB or queued in pending_enrichments.
- Blog compass: for times / guardian / independent, check blog availability
  first (clues.explanation, LENGTH > 30). If with_blog = 0 → STOP and tell the
  user; do not solve blind.
- Honesty over score. FAIL/LOW are fine; a false HIGH is not. If you can't parse
  it, say so.
- Self-check before "done": re-run the work-list query; if it returns rows, you
  didn't finish.
- Role boundary: Claude assists + diagnoses; the USER commits. Never re-solve to
  chase a green light (that optimises for pass-rate, not truth).

**How the post-publish diagnosis works (for context):** the cascade surfaces the
misses; core.diagnose answers "is the material there / which engine came
closest"; that splits each miss into **data** (queue the missing entries),
**signature** (derive the shape from stored assignments via
_cand_from_assignments, tier it, file PENDING-ONLY — safe by construction), or
**engine** (upsert engine_worklist, escalate). Writes allowed: pending-only
catalog rows, engine_worklist, the report log — nothing else. One pass, no loop.

Note: a diagnosis already landed today — memory
container_engine_phrase_blind_indicators (PROVEN 2026-07-17): the container
signature engine never consults phrase-level indicator entries; worklist rows
multiword_container_indicator / container_literal_inner /
container_inner_charade_interior_link. Context for today's container fails.

---

## 1. LAUNCH STATE — live + secured (2026-07-16, all pushed to origin/redesign)

- Site flipped LIVE: nginx maintenance 503 removed (the one `return 503` line in
  /etc/nginx/sites-enabled/cordelia). Public home + /puzzles = 200. Origin
  Cloudflare-only allowlist + SSL intact. Pre-flip nginx backup at
  /root/nginx-backups/cordelia.bak-20260716-prelaunch. See memory
  site_maintenance_mode_2026_05_15 (now marked LIFTED).
- **SECRET_KEY** now read from the environment (systemd Environment= line on the
  droplet); ProductionConfig hard-fails on the dev placeholder. Was the public
  committed dev value in prod. Commit b5b16a91.
- **ADMIN_KEY** rotated off the public "dev-admin-key" default to a strong
  URL-safe value. The VALUE lives in the droplet systemd unit + the user's
  password manager — deliberately NOT written in this doc or any repo file.
- **DB-harvest limiter** tightened: /helper/* reference-DB endpoints now on the
  cross-worker SQLite limiter, 20/min (was a per-worker in-process 60/min that
  reset on restart); clue-corpus endpoints 120/min. Commit 4ab9693b.
- **Cloudflare** = Free plan, still in front; Bot Fight Mode ON. GSC URL
  Inspection returned "URL is available to Google" = Googlebot is NOT challenged
  (the SEO-critical check). If that ever changes, swap Bot Fight Mode for a WAF
  rule (UA googlebot AND cf.client.bot=false → challenge). See
  security_step8_secret_key_from_env.

Security memory START HERE: security_step8_secret_key_from_env.md (full launch +
verification detail).

---

## 2. EXPLANATION DISPLAY — reworked + live (2026-07-16)

The public "full explanation" overlay (puzzle → explanation → full explanation =
POST /wfwfull, rendered by web/wfw_read.py load_breakdown + wfw_full.html). Three
changes, all deployed and verified on the live server via the real click-path:
- Reads TOP-TO-BOTTOM in CLUE word order — every row (definition / source /
  indicator / link) placed by its clue char-position from atom_ids, not grouped
  by role. Commits 317b9b4f (sources) + e34a91b0 (full interleave).
- The CLUE line is PLAIN text (colour lives in the breakdown). Commit c5e1bd2f.
- Colour maps ONLY to the answer: source pieces keep their palette colour
  (matching the answer tiles); definition / indicator / link render plain
  neutral. Commit d8bb040e.
See memory wfw_explanation_clue_order.md for the settled display rules — do NOT
reintroduce clue-word colouring or role colours for def/indicator/link.

---

## 3. OPEN LOOSE THREADS (none block the puzzle review)

- **Admin `/solver` card** (core/wfw_render.py) still renders answer-order in a
  couple of spots; the PUBLIC overlay is correct. Align for consistency when
  convenient (that's the admin view only).
- **Container "leftover letter" fragments** in split-source containers (a bare
  letter run spills, e.g. stray ALLY/ME) — pre-existing, now at least in clue
  order; deeper polish.
- **DB re-upload** was held on the no-current-day rule; user manages DB uploads
  via the DEPLOY page.
- Launch-day checklist tail (documents/LAUNCH_CHECKLIST_2026-07-17.md): styled
  410 page, GSC sitemap resubmit + request-indexing of key pages.

---

## 4. ENVIRONMENT / OPS

- Deploy to the droplet = the dashboard DEPLOY page (the ONLY sanctioned path),
  OR scp the changed files + `systemctl restart cordelia`. Nightly does NOT push.
- Dev server: .venv\Scripts\python.exe web\run_dev.py on :5000. **Reloader is
  OFF** — restart after any .py change; templates auto-reload in dev but are
  CACHED in production (restart the droplet after a template deploy).
- Droplet: root@165.232.46.255, app at /opt/cordelia, gunicorn wsgi:app
  (create_app("production")) on 127.0.0.1:5002 behind nginx. No .env on the
  droplet — env comes from the systemd unit's Environment= lines.
- Admin login: justcordelia.com/?admin=<the rotated key> (value in the systemd
  unit / password manager). Persists ~30 days via the session cookie.
- Keyless SSH to the droplet works from this machine (Bash tool).

---

## 5. RULES THAT BIT (2026-07-16) — read before debugging any "it looks wrong"

- **Reproduce the real path first** (new memory feedback_reproduce_real_path_first):
  a display bug cost ~1 hour because I fixed/verified the WRONG renderer and
  chased cache/DNS/device theories. FIRST get the exact URL + click-path + a
  pasted line; then test the ACTUAL served HTTP output via that path (grab the
  page token, POST it) — not the underlying function.
- **Two renderers for "the same thing":** the inline clue-page CARD
  (core/wfw_render via web.serving.get_card) is DIFFERENT from the /wfwfull
  OVERLAY (web/wfw_read.load_breakdown). Confirm which one the user sees.
- **Verifying a function ≠ verifying the running server.** Fresh-python proves
  the file; only an HTTP request through the real flow proves what gunicorn
  serves. (Dev reloader off; prod caches templates — restart to be sure.)
