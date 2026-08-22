# HANDOVER 2026-08-22 — YouTube pipeline live, CHAPTERS UNSOLVED

> ## RESOLVED later the same day — read this before §"THE NEXT JOB"
>
> **Chapters are a gated feature, not a formatting problem.** Google's feature-access
> page (support.google.com/youtube/answer/9890437) lists "Add chapters" under
> **Advanced** access; the chapters page (answer/9884579) says "If you don't yet have
> access to chapters, apply for access to Advanced features."
>
> Standard = upload. **Intermediate = phone verification -> videos over 15 minutes.**
> **Advanced = valid ID or video verification, or sufficient channel history ->
> chapters, clickable description links, monetisation.** The channel is at
> Intermediate: `longUploadsStatus: eligible` is the phone step, and it was mistaken
> for the whole of verification.
>
> Evidence gathered before concluding it:
> - The descriptions ON YOUTUBE are correct — read back with `videos().list` (1 quota
>   unit) for all four public videos. Real newlines, timestamp at start of line, 0:00
>   first, ascending, 12s apart, 29-33 chapters, last chapter ending exactly at
>   `lengthSeconds` (394 / 346). The text was never the problem.
> - `"decoratedPlayerBarRenderer":{}` — present but EMPTY on every video.
> - **Independent corroboration**: `urlEndpoint: 0` on all four watch pages —
>   justcordelia.com is not linkified in any description. Clickable description links
>   are gated at the same Advanced level. Two Advanced-gated features, both absent.
> - It also explains the §"strongest remaining lead": no AUTOMATIC chapters either.
>   The feature is off for the channel, so neither kind can appear.
>
> The §"chapter length" elimination below was right for the wrong reason — 12s does
> clear the documented 10s floor, but the scene-detection measurement never tested it.
> Hypothesis 1 (API-set descriptions) is WRONG; hypothesis 2 (new channel) was right
> in outline but the mechanism is access level, not history.
>
> **Applied for Advanced features 2026-08-22 — PENDING.** Path is YouTube Studio ->
> Settings -> Channel -> Feature eligibility. **NOT youtube.com/verify**, which is only
> the phone step and was the first thing tried. No code change is needed either way.
> Unknown: whether the four existing videos gain chapters retroactively or need a
> description touch (50 units) to re-parse.

Cold start for the YouTube thread. Read `documents/YOUTUBE_CHANNEL_BRIEF.md` first —
it is the spec and every claim in it carries a file:line. This document is the
delta: what works, and the one job that is open.

## THE NEXT JOB — make chapters render

Nothing else is blocking. The pipeline builds and uploads correctly; chapters are
the only failure.

### What is wrong

YouTube has not created chapters for the video. Not "they are hidden somewhere" —
they do not exist. Proved by reading the watch page's own data:

```
ytInitialData:  macroMarkersListRenderer      absent
                chapterRenderer               absent
                multiMarkersPlayerBarRenderer absent
```

The timestamps in the description are clickable seek links, which is ordinary
description behaviour and NOT chapters. The progress bar is unsegmented and there is
no chapter panel.

### What is already ruled out — do not re-test these

| Suspect | How it was eliminated |
|---|---|
| Chapter formatting | A bare description of nothing but 12 timestamps, 30s apart, plain ASCII titles, also produced no chapters |
| Chapter length | The file's image changes measure at exactly 12.000s apart (ffmpeg scene detection: 9.97, 21.97, 33.97, 45.97…) |
| First chapter not at 0:00 | It is at 0:00 |
| Fewer than 3 chapters | There are 33 |
| Private video | It is now **public** and still has none |
| "Allow automatic chapters" unticked | It is ticked (Studio → Details → Show more) |
| `#` hashtags in the description | Stripped from the scripts before `K7POS9QULlg` was ever uploaded; that video was clean from the start |
| Still processing | `processingStatus: succeeded`, Studio lists SD/HD/4K, and it is now a day old |

### The strongest remaining lead

**YouTube produced no AUTOMATIC chapters either.** Those owe nothing to our
description — YouTube generates them on its own for eligible videos. Their absence
says chapter generation has not run for this video *at all*, which points away from
our text and towards the video or the channel being ineligible.

Two hypotheses, untested:

1. **API-set descriptions may not trigger chapter parsing.** Everything here has been
   written by `videos().insert` / `videos().update`. Test cheaply: edit the
   description **by hand in Studio**, change one character, save, reload the watch
   page. If chapters appear, the fix is procedural, not textual.
2. **A brand-new channel may not be eligible.** No history, no watch time, one public
   video. Nothing to do but wait, but worth knowing before more effort is spent.

A third, weaker: the video is 33 near-identical still frames. It is conceivable
YouTube's chapter machinery expects real motion.

### Before spending anything

An upload costs **1600 quota units of 10,000/day**; a description update costs 50.
Test with updates, not uploads. Yesterday's four uploads used ~6,400.

## What works — do not rebuild it

Five scripts. Full detail in the brief §4b/§4c; memory
[[youtube_pipeline_built]] and [[youtube_channel_auth_setup]].

- `youtube_auth.py` — credential. Channel `UCPUPMydfeAAC7lFTPOMfRFw`, long uploads
  eligible.
- `youtube_capture.py` — one PNG per clue off the clue page, app served in-process on
  loopback (Cloudflare challenges headless Chrome on the live site).
- `youtube_assemble.py` — 2160p frames, intro card, amber call-to-action strip under
  the card with Cordelia in it, chapters.txt + description.txt.
- `youtube_upload.py` — all three papers in one run, ledger-guarded, **private by
  default**.
- `youtube_short.py` — vertical Short, builds but does not upload.

Plus Step 5 in `dashboard/pages/deploy.py` (off by default, gated on `deploy_db`,
cannot fail the deploy) — **verified by reading and syntax only, never exercised in
the Streamlit UI**.

### The daily run

```
.venv/Scripts/python.exe scripts/youtube_upload.py --build
```

Telegraph, Times and Guardian in turn, private, skipping anything filmed or not fully
served. **No day-of-week rule and there must not be one** — Saturday's prize,
Sunday's Prize Toughie / Everyman / Times Sunday all fall out of `classify_puzzle`.

## Loose ends the user owns

- `K7POS9QULlg` is **public** — a correct video of 31324, but a test build. Leave or
  unlist.
- Three superseded private uploads: `ah8yDK_8vIs`, `9cytNyxOsRo`, `3AdX0pTOoNY`.
  Deletable; untouched.
- **The OAuth token expires ~2026-08-28** (app in Testing). Google verification is
  the fix and needs a demo video, which can now be recorded. See brief §4b.
- `@justcordeliacom` handle change never took; still `@justcordelia-m2u`.
- Nothing in this thread is committed to git.

## How this thread went wrong, twice

**Quality.** Three rounds of raising resolution and bitrate against "it looks hazy".
The cause was the player defaulting to 720p; at 2160p the user called it excellent. A
decoded frame had measured 47 dB PSNR against its source at the FIRST attempt — the
evidence that the encode was fine existed before any of the rebuilding. *Check what is
being served before rebuilding what was produced.*

**Guessing.** The first hours were spent relaying a previous handover's assumptions as
fact — including a publish trigger that could not exist and a "decision" about whether
to show the clue text in a video explaining that clue. The user's instruction stands:
state nothing about this system that has not been read in the current session; say "I
don't know" instead of inferring.
