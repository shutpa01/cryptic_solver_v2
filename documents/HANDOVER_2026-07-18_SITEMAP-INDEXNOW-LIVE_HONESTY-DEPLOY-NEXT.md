# HANDOVER 2026-07-18 — SITEMAP + INDEXNOW LIVE; honesty gate COMMITTED (undeployed); SEO = recovery

Cold-start document. Read this, then MEMORY.md. Plain English; **verify against the code
before acting** — memory/handover reflect what was true when written. Everything below is
committed and pushed to **origin/redesign (HEAD = eb72c798)** unless stated otherwise.

---

## 0. ★ STATE OF PLAY — what is LIVE vs COMMITTED vs OPEN

**Git:** branch `redesign`, pushed to `origin/redesign`. Four new commits this session:
- `7f9f2929` sitemap disk-cache fix
- `a77bc42c` IndexNow automation
- `8e9da397` IndexNow incremental (watermark)
- `eb72c798` honesty batch (full clue-type, homophone gate, resolve-from-assignment, **AI-fabrication gate**)

**LIVE on the droplet** (deployed via scp + `systemctl restart cordelia` this session):
- The **sitemap disk-cache fix** (web/routes/seo.py) — clue sitemap went from ~12s to ~0.25s.
- The **IndexNow key file** (web/indexnow.py + the /<key>.txt route in seo.py).

**COMMITTED but NOT deployed to the droplet** — the honesty batch `eb72c798`
(core/wfw_web.py, admin_db.py, store.py, prefill_commit.py, wfw_render.py,
web/routes/clue.py, web/wfw_read.py). **This is NOT a live-site safety gap.** Prefill,
Confirm, hand-solving and the reference-DB harvest ALL run on the LOCAL dev admin (the
/solver mount, admin-gated) — the droplet only SERVES the finished, already-vetted wfw_*
data you upload. So the fabrication gate lives exactly where a fabrication could enter
(local dev) and activates there after a reloader restart (web/run_dev.py; reloader OFF).
The /solver mount does also exist on the droplet (one app, admin-gated), but you confirm
locally and deploy the DB, so nothing fabricated ever reaches the live reference DB. The
ONLY part of this batch that changes what the droplet SERVES is the public clue-type label
(web/wfw_read._wordplay_label, used by the public clue page) — purely cosmetic: it makes
live labels name every mechanism as dev already does. So deploying this batch to the
droplet is OPTIONAL/cosmetic, never a safety requirement.

**NOT in git (intentionally):** `.claude/settings.local.json` (local), `scraper/**/*.json`
(scraped data), `core/_archive/*` untracked dev scripts (gitignored `core/_*`), the handover
docs in `documents/`.

**NEXT, in priority order:**
1. **Your normal DB publish** carries the PRIOR session's data changes (§4) live — back up
   the droplet DBs first. The honesty CODE deploy is OPTIONAL/cosmetic (public clue-type
   label only, per §0) — NOT a safety requirement; fabrication safety is fully handled on
   the local admin where Confirm runs.
2. **Watch Bing** — confirm IndexNow submissions verify (Cloudflare caveat, §2) and that the
   sitemap "temporary processing error" clears now the 12s timeout is gone.
3. **SEO is a recovery, not a build** — see §2. Stability + time. **Never churn URLs again.**

---

## 1. WHAT WAS DONE THIS SESSION

### A. Deploy tooling sped up + core/ decluttered (committed)
- `dashboard/pages/deploy.py` code step now uploads **one scp per directory** instead of one
  per file (was ~330 handshakes = 15 min), and skips underscore-prefixed dev scripts.
- The **83 one-off dev scripts** (`_ab_*`, `_seed_*`, `_diag_*`, `_mine_*`, ...) moved from
  `core/` into **`core/_archive/`** (core 210→127 files). They still run: `python -m
  core._archive._seed_x`. `core/_*` is gitignored so most were untracked; 2 tracked ones
  moved as git renames. Memory: [[deploy-fast-and-core-archive]].

### B. AI-fabrication gate (the session's headline honesty fix; committed, undeployed)
The prefill BUILDER is a headless Claude agent; it can fabricate a letter-source (proven:
`take=R` on clue 10079981 RIA, where the DB only has `take→T`). The mechanical cascade
CANNOT fabricate — only the AI can. The old gate `_build_manual_parse` validated
synonym/abbreviation pieces on GEOMETRY only (do the letters spell the tiles) and stamped
them `source="db"` + queued them for harvest — so leftover-assignment fabrications passed
and would have been written into the reference DB on Confirm.
- **The goal is ENRICHMENT** (AI proposes new vocab → human curates → DB grows), so the fix is
  NOT to block AI proposals. It is provenance: `_build_manual_parse(..., verify_db=True)` (the
  prefill filing, /prefillconfirm, and `_resolve_from_assignment`) now checks each
  synonym/abbreviation against the DB (`admin_db.has_synonym`/`has_substitution`). Not in DB →
  `source="pending"` (renders "provisional"), kept OUT of `db_adds`, queued for review. Confirm
  REFUSES to freeze-pass while a pending synonym/abbreviation piece remains (homophone
  provisionals still pass — mechanism-scoped). The human /hs commit is unchanged (authority).
- Verified via the real functions on RIA. Memory: [[fabrication-gate-prefill-verify-db]].
- **Follow-up (not done):** an unsourced *substitution* is queued as `type='synonym'`, so
  dashboard Accept files it to synonyms_pairs, not wordplay — minor mislabel; a real
  abbreviation still resolves. Proper routing = a `type='substitution'` queue + Accept path.

### C. Sitemap slow-generation fix (committed + DEPLOYED)
`/sitemap-clues-1.xml` took ~12s (renders a WFW card per clue via `is_served`→`get_card`
over 1599 clues) — the likely cause of Bing/GSC "temporary processing error". Fix in
web/routes/seo.py: shared **on-disk** cache (tempfile/cordelia_sitemap) with
stale-while-revalidate — one worker builds, all serve it; URL set byte-identical (287628
bytes / 1599 URLs). **On disk, not in-memory** — the first in-memory attempt passed dev but
FAILED on the droplet (gunicorn multi-worker; each cold worker still ~10s). Caught by testing
on the real target. Memory: [[sitemap-slow-generation-fix]].

### D. IndexNow automation (committed; key file DEPLOYED; backfill done)
Bing acts on URL submissions (unlike Google's ignored request-indexing), so we notify it.
- `web/indexnow.py` — public KEY `c4c9e81fbf21629f7835000e0e13dbd9` (NOT secret) + `submit()`.
- Key file live at `justcordelia.com/<KEY>.txt` (route in seo.py; Cloudflare only intercepts
  /robots.txt so this passes through).
- `scripts/indexnow_notify.py` — computes served clue+puzzle URLs via the app's OWN
  is_served / served_puzzle_numbers (never announces a 410). DEFAULT = **incremental**:
  announces only what's served since a watermark (logs/indexnow_watermark.txt, gitignored),
  advancing only on success. `--all` backfill, `--days N` window, `--init` baseline.
- `dashboard/pages/deploy.py` Step 4: fires the notify after a DB deploy (never fails the
  deploy). **So the daily flow is: finish a puzzle → deploy the DB → Bing gets just that
  puzzle, each URL once.** Ran `--all` once (1625 URLs, HTTP 200); watermark baselined.
  Memory: [[indexnow-automation]].

---

## 2. SEO — the real situation (diagnosed this session; NO code fix will re-index)

- The site was INDEXED, then **deindexed from thousands of pages to ~36**. Cause: a URL
  **remove-then-re-add churn** (from PRIOR bad assistant advice) plus a ~2-month **503**
  maintenance period. Confirmed: GSC **Manual actions = clean**, no 503/noindex now, internal
  linking server-side sound, served pages return clean 200, legacy returns 410 by design.
- **This is a trust recovery: stability + time.** There is no switch. The sitemap fix +
  IndexNow remove obstacles to DISCOVERY; they do not re-index. **NEVER propose URL reduction
  or churn the sitemap/URL set again** — that is exactly what caused the damage
  ([[feedback-never-propose-url-reduction]], [[feedback-no-sitemap-size-flipflop]]).
- The 410 model (week-only, no legacy) is deliberate and CORRECT — those clues will never be
  served. Do not touch it.
- **Bing set up** (via GSC import): site verified, `sitemap.xml` submitted (index → 3 child
  sitemaps), `sitemap-clues-1.xml` submitted → **1.6k discovered**.
- **CAVEAT to watch:** IndexNow verifies ownership by fetching the key file THROUGH Cloudflare.
  Submissions returned 202 (accepted, verifying). If Bing WMT's IndexNow report shows failures,
  it's Cloudflare Bot Fight Mode blocking the verifier's key-file fetch — fix with a Cloudflare
  rule allowing the key file / IndexNow.
- Growth advice retracted honestly: do NOT chase backlinks from the incumbent solver blogs
  (Big Dave's, fifteensquared — they're competitors); Reddit links are nofollow + anti-promo.

---

## 3. DEPLOY steps (user drives; assistant cannot click the DEPLOY page)

The DB publish is the normal daily step. The honesty CODE deploy is OPTIONAL (cosmetic
public clue-type label only — §0); it is NOT required for safety.

1. **Back up both droplet DBs first** (no undo on overwrite):
   `ssh root@165.232.46.255 "cp /opt/cordelia/data/clues_master.db{,.bak-20260718} && cp /opt/cordelia/data/cryptic_new.db{,.bak-20260718}"`
2. Upload the DBs via the dashboard **DEPLOY page** — this makes ALL of today's content live,
   including the PRIOR session's data changes (§4). Confirm each puzzle is in the state you
   want public first. Restart is handled by the flow; IndexNow fires automatically after
   (incremental).
3. (Optional) Deploy the honesty CODE via the DEPLOY page if you want the public clue-type
   labels to name every mechanism. This session's live files (seo.py, indexnow.py) are
   already on the droplet by hand-deploy; the honesty batch is not.

Note: seo.py + indexnow.py were deployed by hand this session (scp + restart). The origin's
robots.txt is served by **Cloudflare** at the edge (not the Flask route) — verified; the
sitemap deploy did NOT change robots.

---

## 4. STILL-OPEN DATA TO-DOs carried from the PRIOR session (not touched this session)

From HANDOVER_2026-07-17_HONESTY-FIXES — verify these are still wanted before deploying the DB:
- **3 tentative homophone pairs to APPROVE** in pending_enrichments: over~ova, re-sinned~rescind,
  dough~deau. Until approved they render provisional.
- Invalid clues to click through (10079943 CENTRE OF GRAVITY, 10079976 ABLE SEAMAN) — confirm
  no wrong parse shows.
- RIA (10079981) stays a FAIL (take=R is fabrication; not in the DB). It is currently a frozen
  INVALID — leave it.

---

## 5. PRINCIPLES RE-AFFIRMED THIS SESSION (read before acting)

- **Honesty over passes / over speed.** The fabrication audit is the whole point: a true FAIL
  beats a false solve. Do not add vocab to the DB to force passes.
- **Verify on the REAL target.** The sitemap in-memory cache passed dev and failed on the
  droplet's multi-worker gunicorn — only a droplet test caught it. "It works" needs proof
  through the real path.
- **No bluffing; say when you don't know.** The SEO exchange went badly when the assistant
  offered generic playbook advice (backlinks, Reddit, competitor blogs) that didn't survive
  the user's real constraints. On growth strategy against entrenched competitors: the honest
  answer was "I don't have a credible plan," not invented tactics.
- **SEO scar tissue is binding:** never churn URLs / sitemap size; the deindexing was
  self-inflicted by exactly that.
