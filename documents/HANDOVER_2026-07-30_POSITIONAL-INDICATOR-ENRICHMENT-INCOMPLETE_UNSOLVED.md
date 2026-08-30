# COLD HANDOVER — 2026-07-30 (late)

**Read this as an honest account from a session the user judged a failure.** Three hours, and the
only concrete outcome was hand-solving four individual clues. The CLASS problem the user actually
wants solved is NOT solved. Labels below are VERIFIED vs NOT. Do not repeat this session's pattern:
guess-a-diagnosis-and-claim-success. TRIAGE FIRST, test the REAL running process, prove before claiming.

---

## 0. THE UNSOLVED PROBLEM (this is the job — start here)

Cryptic clues whose reading uses a **charade_positional** indicator (e.g. "goes after", "followed by",
"deposited after") will not Confirm and each needs a MANUAL per-row Approve where the human picks the
direction (after/before). This recurs on clue after clue. It is NOT solved for the class.

**Root, in the system's own terms (the user corrected my wrong framing — do not say "infer the
reading"; the system SUGGESTS ENRICHMENTS):** the enrichment SUGGESTION for a positional indicator is
INCOMPLETE. When "goes after" is queued it is stored as (type='indicator', word='goes after',
letters='charade_positional') with **no direction**. Compare a SELECTION indicator: the /hs reading
records its sub-type (`isub`, e.g. 'alternate') — but the positional reading records NO `isub` at all
(VERIFIED: INSTRUMENTALIST and EAGLE assignments have `{"role":"indicator","itype":"charade_positional"}`
with no `isub`). So the suggested enrichment cannot carry after/before, `add_indicator` refuses a
directionless charade_positional, and the user must supply it by hand every time.

**The fix must live where the enrichment is SUGGESTED** — the prefill that builds the /hs assignment and
calls `queue_indicator`. I did NOT read/locate that prefill path. That is the next thread's first job:
find where positional indicators are tagged and queued, and make the suggestion carry the direction
(so approving is one complete click, or the piece auto-backs). `queue_indicator` (core/pending_store.py)
currently stores only the type in `letters` — no subtype column — so this likely needs the suggestion to
encode/carry the sub-type end to end. DO NOT bolt on confirm-time "inference"; that is not the model.

Scope check (VERIFIED): only 23 pending prefill clues exist across all sources. pending_enrichments has
11,054 definitions (mostly old backfill), 206 indicators (188 old anagram backfill). The live positional
backlog is tiny right now (e.g. "deposited after" → RUING is one). So this is about stopping the RECURRENCE
on new puzzles, not clearing a big backlog.

---

## 1. WHAT WAS ACTUALLY DONE (code — UNCOMMITTED, working tree only; HEAD = 0b0a59d0)

Three files modified, none committed, nothing deployed. Live site unaffected.

- **core/admin_db.py `has_indicator`** — for `wordplay_type=='selection'`, now returns True if ANY
  contiguous sub-run of the tagged span maps to a selection rule via `selection_indicators.SUBTYPE_RULE`
  (covers the parts↔selection taxonomy AND phrase→sub-word, e.g. 'end'=parts/last, 'regularly selected'
  →'regularly'). Other types keep exact match. VERIFIED via test_client + running server + 8-case
  pos/neg test. This genuinely fixes the SELECTION false-negative class.
- **core/wfw_web.py `_enrich_row`** — the per-row indicator Approve now renders a sub-type dropdown
  (`_IND_SUBTYPES` for the queued type). Before, it had no sub-type field, so approving a selection/
  positional indicator silently failed. VERIFIED.
- **core/wfw_web.py `/enrich` route** — delete the pending row ONLY on success (`msg` startswith
  Added/Already/Approved); a failed add no longer eats the queue row. VERIFIED.
- **core/wfw_web.py `_pending_add_form` + `approveall_route`** — Approve-all now (a) recovers an
  indicator's sub-type from the clue's /hs assignment (itype→isub) where present, (b) deletes+counts a
  row ONLY on a successful add, (c) reports "N need a sub-type — approve individually" honestly instead of
  the old false "Approved N". VERIFIED on running server. NOTE: this does NOT help positional indicators
  whose assignment has no `isub` (the unsolved problem above) — they still fail Approve-all and need the
  per-row pick.
- **web/run_dev.py** — `_kill_stale_listeners(5001)` + `_wait_port_free(5001)` before `app.run`, so a
  restart kills any existing 5001 listener and comes up as the SOLE server. VERIFIED against the OS
  (launched while 2 stale servers ran → both killed/dead → one listener). This closes the recurring
  multi-stale-server trap that made half this session's "it works" claims false. See memory
  [[dev_server_multiple_listeners_recurring]].

## 2. DB / STATE CHANGES I MADE (local, shared DBs — you should know these)

- Froze to status='pass': BANDANA 10082098, TREASURER 10082178, INSTRUMENTALIST 10082194, EAGLE 10082185.
- Added to data/cryptic_new.db `indicators`: 'end'/selection-last, 'regularly selected'/selection-alternate,
  'followed by'/charade_positional-after, 'goes after'/charade_positional-after.
- **TIMES 29608 is now 30/30 pass** (VERIFIED). That puzzle's clues no longer block. NOT published/deployed.
- Dev server: single V2 listener on :5001 (was PID 96072, V2 venv). run_dev.py self-cleans now.

## 3. WHAT IS NOT DONE
- The positional-indicator class problem (§0). This is the real ask; it is untouched at the source.
- Nothing committed. Nothing deployed. The live site has NONE of §1. Puzzles are NOT published.
- The prefill-suggestion code path was never read.

## 4. BEHAVIOURAL WARNINGS (why this session failed — do not repeat)
- **Test the EXACT running process the user hits.** I repeatedly "proved" fixes via `test_client`/fresh
  `python -c` (a separate process) while the user's browser hit a STALE server running old code — there
  were MULTIPLE dev servers on 5001 at once. Every "it works now" that followed was false. After any
  restart: `netstat -ano | grep :5001 | grep LISTENING` → confirm EXACTLY ONE, V2 venv. Never trust curl 200.
- **Triage before diagnosing.** The real remaining blocker (EAGLE) was found only by COUNTING the puzzle's
  pass/fail (29/30). I should have led with that, not chased individual clue IDs handed to me.
- **Don't guess-and-stick.** I fixed selection, then treated every later clue as "the same", missing that
  Approve-all (the user's actual button) and positional indicators were different causes.
- **Reference-DB enrichment is how vocab enters** (user: "we suggest enrichments") — NOT hardcoded lists,
  NOT confirm-time inference.

## 5. ENV
- Dev: `web/run_dev.py` on :5001, V2 venv `.venv\Scripts\python.exe`, reloader OFF (full restart per .py
  change — now self-kills stale listeners). Admin session via `?admin=<ADMIN_KEY>` (config ADMIN_KEY).
  Solver UI mounted at `/solver/*`, admin-gated (web/solver_mount.py). Clue confirm route:
  POST /solver/prefillconfirm (fields: only, id); Approve-all: /solver/approveall; per-row: /solver/enrich.
- data/clues_master.db (clues, wfw_*, pending_enrichments); data/cryptic_new.db (reference: indicators,
  synonyms_pairs, wordplay, definition_answers_augmented, ...). Both gitignored.
- Memory: [[indicator_gate_selection_taxonomy_rootcause]], [[dev_server_multiple_listeners_recurring]].
