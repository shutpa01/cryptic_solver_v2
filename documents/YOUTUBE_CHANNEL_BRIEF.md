# YouTube channel — brief

Written 2026-08-21. Supersedes `HANDOVER_2026-08-21_YOUTUBE-CHANNEL-BRIEF_DECISIONS-OPEN.md`,
whose "five open decisions" were mostly not decisions (see §6).

Nothing is built. Nothing is committed.

Every claim in §3 was read from the code in the session that wrote this and carries a
file:line. Anything not cited is either labelled as the user's instruction (§2) or named
as unknown (§4). Nothing here is inferred from a previous handover.

---

## 1. What it is, and to what end

A daily YouTube video that steps through the WFW clue pages of a puzzle, exactly as the
site already renders them — captured, stitched in clue order, one chapter per clue, the
puzzle number in the title, uploaded automatically.

The end is discovery, on two channels:

- **Puzzle-level search.** People search for a specific day's puzzle by number. The
  volume per puzzle is small and the intent is total.
- **Clue-level search.** Per-clue chapters make one upload into ~28 timestamped,
  independently indexable targets, which YouTube surfaces as "key moments" in Google
  video results. This is the same long-tail demand the clue pages already serve, on a
  surface Google indexes independently of the site — which matters given the Bing
  situation.

Plus a daily Short off the same pipeline, for people who are *not* searching. That is
the only part of this aimed at people who don't already have a query.

No new content is created. No second renderer is written. The video is a re-presentation
of pages that already exist and have already passed review.

### What it will not do

YouTube description links are nofollow. This buys no link equity for the site. The return
is referral traffic, branded search, and a second indexable surface. Do not budget for a
ranking lift on the site's own pages.

---

## 2. Requirements (user, 2026-08-21)

1. Per-clue chapters, to leverage clue-level searches.
2. A daily Short, for people who are not searching.
3. The upload runs **as part of the dashboard deploy** — not the nightly run.
4. Extend beyond Telegraph to the other two publications. Served sources are telegraph,
   times, guardian (`web/serving.py:27`), so this reads as Times and Guardian.
5. A **permanent on-screen banner** telling viewers to SEARCH for justcordelia.com,
   where the puzzle can be solved with a range of built-in hints. On-screen, not in the
   description — the description link is nofollow and a search is a branded query.

Settled by the brief itself, not open: every clue (it cycles the puzzle), and the clue
text on screen (there is no explanation without the clue — the site already publishes it).

---

## 3. Verified mechanics

### 3.1 The frame source is a page that already renders complete

The WFW page is the clue page, route `/clue/<slug>` (`web/routes/clue.py:286`).

The card is fetched server-side (`web/routes/clue.py:337`, `get_card`) and printed
inline into the template (`web/templates/clue.html:139`, with its CSS at `:133`). It is
**not** revealed by JavaScript. The htmx on that page is the admin edit overlay
(`web/templates/clue.html:278`, `hx-get /admin/edit/...`).

Consequence: a headless capture of the clue page gets a complete frame on load. No
clicking, no waiting on progressive hint steps.

A clue with no card 410s (`web/routes/clue.py:339-340`), so a capturable page and a
served page are the same thing.

### 3.2 The publish gate is a status gate, not a date

`puzzle_is_served` (`web/serving.py:143-171`): a puzzle is public only when **every** one
of its clues has a `wfw_solve` row with status `pass`, **or** status `invalid` with a
non-empty reviewer comment. One unsolved clue hides the whole puzzle at its stable URL,
and the page returns when the last clue is solved.

`served_puzzle_numbers` (`web/serving.py:174-...`) is the same rule in bulk, one query.

This is the answer to "how can it publish before prefill has been checked" — it cannot.
Fully-served *is* reviewed. A video can only ever be made of a puzzle that has already
passed, by construction.

The puzzle page itself enforces the same gate (`web/routes/puzzle.py:54-55`).

### 3.3 The trigger is the deploy button

`dashboard/pages/deploy.py` is a Streamlit page with one button (`:116`). Manual, not
scheduled. Steps, in order:

| Step | Line | Gate |
|---|---|---|
| Upload code | `:121` | `deploy_code` |
| Upload clues_master.db + cryptic_new.db | `:233` | `deploy_db` |
| Sync scraper grid JSONs | `:279-299` | `deploy_db`, never fatal |
| Restart cordelia | `:302` | — |
| Warm sitemap cache | `:331` | never fatal |
| IndexNow notify | `:361` | `deploy_db`, never fatal (`:380`) |

The IndexNow step is gated on `deploy_db` because that is the moment new content actually
becomes live. A video upload belongs in the same position, on the same gate, at the end
where a failure cannot take the deploy down.

### 3.4 "Which puzzle today" is already solved, and the pattern is reusable

`scripts/indexnow_notify.py` answers exactly the question a video step has to answer.

`collect_puzzle_urls` (`:79-130`) takes `served_puzzle_numbers()` and builds each fully-
served puzzle's URLs. `main` (`:183`) subtracts a **ledger** of puzzles already announced
— `logs/indexnow_state.db`, table `sent_puzzle(source, puzzle_number, sent_at)`
(`:42`, `:56-58`). Records only on success (`:220-222`), so a failure retries next run.

The video step is the same shape with its own ledger: fully served, not yet filmed → film
it. Puzzle-level, one row per puzzle, recorded only after a successful upload.

Note the design rationale stated at `:8-16` — a puzzle is tracked rather than a clue
because a puzzle is only deployed once every clue is solved, making it a complete stable
unit. That reasoning transfers to video unchanged.

### 3.5 Clue order comes free

`web/routes/puzzle.py:35` calls `get_puzzle_clues(source, puzzle_number)`; `:84-87` splits
it into across then down. That is the order the video should follow.

---

## 4. What does not exist

- **ffmpeg** — not on PATH. Nothing can be assembled without it. Installing it needs
  approval. **This is the next blocker.**
- **No video library** in the venv: no moviepy, imageio, opencv, or Pillow.
- **No capture, assembly or upload code** — only the auth step below exists.

Present and usable:

- **selenium 4.40.0** in the venv (no playwright). The scraper already drives Chrome with
  a profile at `scraper/telegraph/.chrome_profile`.
- **google-api-python-client 2.194.0**, **google-auth**, **google-auth-httplib2**,
  **google-auth-oauthlib 1.4.0** (installed 2026-08-21 with approval; it was missing,
  which means `impressions/search_console_report.py` could not have run before that).

## 4b. The channel and the upload credential — DONE 2026-08-21

Channel `UCPUPMydfeAAC7lFTPOMfRFw`, title justcordelia, handle `@justcordelia-m2u`.
Phone-verified: the API reports `longUploadsStatus: eligible`, so the 15-minute cap is
lifted and a full puzzle walkthrough will upload.

`scripts/youtube_auth.py` mints and verifies the credential. It prints which channel the
token controls, which is the failure worth catching — a token that authenticates but
points at the project owner's own channel would upload silently to the wrong place.

Two accounts are involved and they are not interchangeable. Cloud project
`cordelia-493208` belongs to shuterpaul@gmail.com and holds the OAuth client
(`impressions/credentials.json`, shared with the Search Console script). The channel
belongs to justcordelia.com@gmail.com. The client belongs to the project, but the
consenting account decides the channel — so the token is minted against shuterpaul's
client while signing in as justcordelia. No second Cloud project was needed.

**Live constraint.** The app is in Testing with justcordelia as a test user. Google
denies sensitive YouTube scopes to non-test users until the app is verified, and
"In production" does not help — while in production the test-user list is not in force at
all. A token minted in Testing **expires after seven days**. Fine for building, fatal for
an unattended daily upload.

Verification therefore has to happen, and needs: app homepage `https://justcordelia.com`,
a privacy policy (BUILT and deployed 2026-08-21 — `/privacy`, `web/routes/browse.py`,
`web/templates/privacy.html`, footer link in `base.html`), domain ownership (already
proved via Search Console under shuterpaul), a written scope justification, and a demo
video. The demo video cannot be recorded until the uploader works, so verification is
submitted AFTER the pipeline is built.

---

## 4c. The pipeline — BUILT 2026-08-21

Four scripts, each doing one thing. All output lands in `logs/youtube/` (gitignored,
`.gitignore:130`).

**`scripts/youtube_auth.py`** — mints and verifies the credential (§4b).

**`scripts/youtube_capture.py`** — one PNG per clue, in the puzzle's own order.
Serves the Flask app in-process on a loopback port and drives headless Chrome at
that, because Cloudflare challenges headless Chrome on the public site (proved: the
probe got "Just a moment..." and no card). Same templates, same card — not a second
renderer. Captures the clue panel only; everything after the card inside the panel is
hidden at capture time in the captured browser, nothing on the site is touched.
Clues whose own page 410s — the "See N" continuation stubs, which count as solved for
the puzzle gate but have no card — are skipped and named. Writes `manifest.json`.

**`scripts/youtube_assemble.py`** — pads each panel onto a fixed 1920x1080 canvas
(panels vary 1230–1740px tall, so scaling-to-fit would make the card breathe between
clues), burns in the banner, encodes, and writes `chapters.txt` / `description.txt`.
12 seconds per clue by default; **it refuses anything under 10**, because YouTube
renders no chapters at all if any chapter is shorter, and chapters are the whole
per-clue argument.

**`scripts/youtube_upload.py`** — uploads once per puzzle, ledger in
`logs/youtube_state.db`, same shape and rationale as the IndexNow ledger. Title and
description vary by puzzle type, which is read from `web/models.classify_puzzle`
(`web/models.py:136-157`) rather than from the calendar — so Saturday's prize and
Sunday's toughie get their own titles, and a catch-up upload on a Tuesday still gets
them right. **Private by default.**

**`scripts/youtube_short.py`** — one clue as a vertical 1080x1920 Short: the clue
alone first (the card hidden in the captured browser — a question with no answer on
screen), then the identical panel whole. Both frames anchored at the SAME top edge,
not centred, so the clue does not jump up the screen at the moment someone is reading
it. Default clue is the one with the richest breakdown (tallest captured panel),
deterministic; `--clue "9 Down"` overrides. **It does not upload** — building is safe
to automate, publishing is a decision.

Wired into `dashboard/pages/deploy.py` as Step 5, behind an off-by-default checkbox,
gated on `deploy_db` and never able to fail the deploy — the same treatment IndexNow
gets, for the same reason.

### Presentation, after the SECOND review (2026-08-21) — still not settled

The first round of changes was judged "very little improvement". What followed:

- **2160p, captured at 4x** (`--resolution 2160p --scale 4`, 45 Mbps). The limit was
  never the encode — a decoded frame measured **47 dB PSNR** against its source PNG,
  which is visually lossless. The limit is what YouTube gives back: it allocates a
  far richer encode to a 4K source, and even its 1080p rendition of a 4K upload beats
  a native 1080p one.
- **The call-to-action strip moved under the card** and turned amber, with Cordelia's
  own picture (`web/static/cordelia.jpg`) in it. Pinned to the frame edge in dark
  navy it read as browser furniture and was never looked at.
- **Cordelia is on every frame**, in that strip, and large on the title card. The
  assembler refuses to build if her picture is missing.

**A trap to know about**: the frame is 16:9 and the card is roughly 1.4:1, so the
card is HEIGHT-limited. Raising the output resolution raises absolute detail but does
NOT make the text larger relative to the frame — that is fixed by the aspect ratio.
Making the text bigger on screen would mean showing less per frame (splitting tall
clues across two frames), not a bigger canvas.

**Timing**: a 6.5-minute 4K upload sits in `processingStatus: processing` for a long
while, and until the high renditions exist YouTube serves a low one. Judging quality
before processing finishes judges the wrong thing.

### THE QUALITY COMPLAINT WAS THE PLAYER, NOT THE PIPELINE

Three rounds were spent on this. The answer: **YouTube's player was defaulting to
720p**. Forced to 2160p the user called it excellent. The pipeline was never the
problem — a decoded frame measured 47 dB PSNR against its source PNG at the very first
attempt.

The lesson is not about video. It is that "the output looks wrong" was assumed to mean
"we produced it wrong", and three fixes were shipped against that assumption before
anyone checked what was actually being *served*. Check the delivery before rebuilding
the thing delivered.

The 2160p upload is still worth keeping: most viewers sit on Auto, and Auto off a 4K
source is better than Auto off a 1080p one.

### DAILY RUN — all three papers, one command

`scripts/youtube_upload.py` with no `--source` runs telegraph, times and guardian in
turn (`--sources` overrides). Each is independent: a source that errors cannot stop
the others, and a source with nothing served and unfilmed just says so.

**There is no day-of-week rule and there must not be one.** The user's schedule —
Mon-Fri DT/Times/Guardian cryptics, Saturday DT Prize and Times, Sunday DT Prize
Toughie / Guardian Everyman / Times Sunday — falls out of `classify_puzzle` and the
serving rule by itself. Whatever is served and unfilmed gets filmed.

Verified 2026-08-21: all three papers have all of their last 12 puzzles served.
Three uploads a day is 4,800 of the 10,000 daily quota. Adding a Short each would
reach 9,600 — at the ceiling, so Shorts need a subset or a quota rise.

**Privacy is `private` by default**, in both the script and the deploy dropdown. The
user will review the first day's batch by hand before considering automatic release.

### STILL OPEN — chapter navigation does not render

The description's timestamps are clickable jump links, and 33 chapters are present and
correctly formatted (first at 0:00, all ≥10s, ascending). But the player's scrubber
shows **no chapter segments**, and the user reports no visible navigation.

Ruled out: too few chapters, sub-10-second chapters, a first chapter not at 0:00.
Tried and did NOT fix it: removing `#` from the description, which YouTube was turning
into hashtag links (worth keeping anyway — hashtags get promoted above the title).
Being tried: **unlisted rather than private** — the one variable not yet eliminated.

Do not claim this is fixed without seeing segments on the scrubber in a real browser.
The automated Chrome could not play the video at all, so screenshots of it prove
nothing either way.

### Presentation, after the first review (2026-08-21)

The first cut was called "not production level" and it was right. Four changes:

- **1440p, not 1080p, at CRF 16 on the `slow` preset.** The whole video is thin
  coloured text on white, which is what chroma subsampling and a low bitrate destroy
  first — and YouTube gives a 1440p stream a markedly higher bitrate than a 1080p one.
  `--resolution` and `--crf` expose both.
- **Banner moved to the TOP.** At the bottom it read as a footer and the eye never
  went there. It is also proportional to the frame now, with the lead word in amber.
- **An opening title card** — paper, puzzle type, number, "Every clue explained",
  and the full date — held for 10 seconds as the chapter that starts at 0:00. A video
  that opens cold on clue 1 tells a searcher nothing about whether they are in the
  right place.
- **Tighter margins**, so the card owns the frame instead of floating in it.

The card is always height-limited in 16:9 — it is wider than tall, but not by 16:9 —
so there is space either side that cannot be filled without stretching or cropping the
card. The card is the product; the space stays.

The banner's type is sized from its HEIGHT, so a narrow one overflows: at 1080 wide
the tail wrapped and "SEARCH" was clipped. Below 1400px wide the tail is dropped.

### Proof, 2026-08-21

Telegraph 31324, 32 clues, none skipped. Video 6:24, 1080p, 5.2 MB, 32 chapters
starting at 0:00. Uploaded private as `ah8yDK_8vIs`; the API confirms title,
`privacyStatus: private`, category 27, `madeForKids: false`, `PT6M24S`, and the
chapter list intact in the description.

### Mistakes worth not repeating

**The archive drain.** The upload step originally took "most recent puzzle not in the
ledger", so a second run uploaded the *previous* day's puzzle unbidden — 51 served
puzzles are unfilmed, and on a deploy hook that is a silent quota drain. There is now
an age guard (`--max-age-days`, default 1, measured against the newest served puzzle
rather than today) which names what it holds back. `--backfill` lifts it deliberately.
`3AdX0pTOoNY` (31323) is the accidental upload — private, harmless, undeleted.

**Silent-size failures.** The banner was first rendered from a `data:` URL, which
Chrome truncates at the first `#` — and the stylesheet is all hex colours. It produced
a blank 1898x32 strip that overlaid onto nothing and no error was raised. It now
renders from a file and the script *asserts the banner is exactly 1920x130* before
building any frame. Same class of bug: the concat demuxer ignores the last entry's
duration, so the usual repeat-the-last-image fix silently made the video 12 seconds
longer than the chapters claimed; it is now cut to the computed total.

**`-shortest` truncates a concat of stills.** A 20-second Short came out 6.03s —
exactly the teaser's duration. With `-t` already bounding the output and the silent
audio track infinite, `-shortest` had nothing to contribute and quietly cut the video
at the first image. Removed from both the assembler and the Short. The long-form video
happened to survive it, which is precisely why it was worth removing there too.

**The ledger's `INSERT OR IGNORE` lied.** A `--force` re-upload printed "recorded"
while leaving the row pointing at the superseded video. It is an upsert now.

**A transient capture miss.** Clue 21 returned `no-panel` from Chrome on a page that
serves 200 with both the panel and the card present, and succeeded on reload. There is
one retry, with the URL and page title printed; a miss that survives the retry still
stops the run, because a wrong frame is worse than no video.

### Quota — the next thing that will break

A `videos.insert` costs 1600 units against a default 10,000/day project quota:
about **six uploads a day for the whole project**, shared with anything else using it.
Three publications plus a Short each is already at the ceiling. The project's real
quota has not been checked.

## 5. Open

1. **Narrated or silent.** Watch time is YouTube's ranking signal, but narration is a
   second build and a quality risk. Not decided.
2. **Copyright posture.** A complaint against a web page is a page; against a channel it
   is a strike, and three strikes ends the channel. This is not a question about whether
   to show the clue — it is a question about whether to run the channel on a corpus we
   already publish. Needs a decision on risk appetite, once, not per-feature.
3. **Google verification submission** — see §4b. Not blocking the build, but the seven-day
   token expiry means it must be done before the channel can run unattended.
4. **Quota.** YouTube Data API upload cost per video against the default daily quota, for
   three publications a day plus Shorts. Not yet checked.
5. **The handle.** `@justcordeliacom` was free on 2026-08-21 and matches the domain the
   on-screen banner tells viewers to search for. The change has not taken — the API still
   reports `@justcordelia-m2u`.

---

## 6. What previous threads got wrong — do not repeat

- **The nightly run is not the trigger.** An earlier draft of this brief said publish
  "after the nightly run". A puzzle is not publishable until every clue has passed review
  and the deploy has put it live. The deploy button is the trigger (§3.3).
- **"Show the clue text, or answer and breakdown only?"** was posed as a product
  decision. It is not one. The clue is the thing being explained; there is no product
  without it, and the site already publishes it. It arrived as a copyright worry and
  should have stayed one (§5.2).
- **"Every clue or a subset?" and "DT only or all three?"** were already answered by the
  brief and by the user's own requirements.
- **A warning about writing a second WFW renderer** was raised from memory rather than
  code, and withdrawn. The clue page renders complete server-side (§3.1); there was never
  a second renderer in question.

The common fault: relying on a previous document instead of reading the system. Every
factual claim about this codebase must carry a file:line read in the current session, or
be stated as unknown.
