# Google OAuth verification — submission pack

Written 2026-08-22. The token minted in Testing expires ~2026-08-28
(`YOUTUBE_CHANNEL_BRIEF.md` §4b: "A token minted in Testing expires after seven
days"). Verification is what replaces it. This document is the submission, not a
summary of it — the justification text below is meant to be pasted.

Requirements are Google's, from support.google.com/cloud/answer/13464321, read
2026-08-22. **This is a separate process from the YouTube channel's Advanced
features application** (that one is about chapters — see
`HANDOVER_2026-08-22_YOUTUBE-PIPELINE-LIVE_CHAPTERS-UNSOLVED.md`). Do not conflate
them; they were conflated once already.

---

## 1. Where each requirement stands

| Requirement | State | Evidence |
|---|---|---|
| Homepage on a verified domain, describing the app, not login-only | **MET** | `https://justcordelia.com` |
| Domain ownership proved | **MET** | Search Console under shuterpaul (brief §4b) |
| Privacy policy in the homepage's domain, linked from homepage AND consent screen | **MET** | `/privacy`, deployed 08-21. `web/templates/privacy.html:57-84` is a "Use of Google account data" section covering what is requested, that it is one account and never a visitor's, where the credential is stored, revocation, and Limited Use compliance. **Check the consent screen links to this exact URL.** |
| Scope justification | **DRAFTED — §3 below** | needs the §2 decision first |
| Demo video | **NOT DONE — shot list in §4** | can now be recorded; the pipeline works |

Two things to verify in the Cloud console before submitting, both common rejections:

1. **App name must match the homepage branding.** The consent screen's app name has
   to be the same product as `justcordelia.com` presents. If the project is still
   showing something like `cordelia-493208`, change it to Cordelia first.
2. **The privacy policy link on the consent screen must be the same URL** as the one
   linked from the homepage — `https://justcordelia.com/privacy`.

---

## 2. Scopes — DECIDED 2026-08-22: Option A, narrow. DONE.

The scope list is now, in both `scripts/youtube_auth.py` and
`scripts/youtube_upload.py`:

    https://www.googleapis.com/auth/youtube.upload
    https://www.googleapis.com/auth/youtube.readonly

Verified after the change: the two lists are identical, the full `youtube` scope
appears in neither, and `youtube_auth.py --verify` still reads the channel back
(justcordelia / UCPUPMydfeAAC7lFTPOMfRFw, long uploads allowed) — the old token is a
superset so it keeps working until re-minted. `youtube_upload.py --dry-run` runs
clean. **Use the §3 justification as written; ignore the Option B variant below,
kept only as the record of what was weighed.**

The consequence, stated once so nobody trips on it: `videos.update` is no longer
covered. Editing an uploaded video's title or description is a Studio job, by hand.

The reasoning that produced that decision follows.

Requested until 2026-08-22:

    https://www.googleapis.com/auth/youtube.upload
    https://www.googleapis.com/auth/youtube          <- full read AND write

What the code actually calls — the complete list:

| Call | File | Narrowest scope that covers it |
|---|---|---|
| `channels().list(mine=True)` | `youtube_auth.py:74` | `youtube.readonly` |
| `videos().insert` | `youtube_upload.py:273` | `youtube.upload` |

**Nothing in the committed code needs the full `youtube` scope.** Google's rule is
explicit: "If the requested scope(s) goes beyond the usage needed, you will be
directed to request a narrower scope." As it stands this submission invites exactly
that, and the round trip costs days we do not have before the 28th.

But there is a real counter-weight. `videos().update` — the 50-unit description
touch that may be needed to make chapters re-parse once Advanced access lands —
requires `youtube`, not `youtube.upload`. So:

- **Option A, narrow**: request `youtube.upload` + `youtube.readonly`. Fastest path
  through review. Description edits then have to be done by hand in Studio, which is
  a handful of clicks, done rarely.
- **Option B, keep `youtube`**: justify it on managing metadata of the app's own
  uploads. Consistent with what the privacy policy already published —
  `privacy.html:62` says "upload and manage videos on that one channel". Slower if
  a reviewer pushes back.

**Recommendation: Option A.** The only thing it costs is doing a rare description
edit by hand, and the deadline is the binding constraint. Note that changing the
scope list means re-minting the token (`youtube_auth.py`), which is also the run
that gets filmed for the demo — so make this decision BEFORE recording §4.

---

## 3. Scope justification — paste-ready

> Cordelia (justcordelia.com) publishes explanations of cryptic crossword clues. The
> application automatically assembles a short video walkthrough of each day's puzzle
> and uploads it to a single YouTube channel that I own and operate
> (UCPUPMydfeAAC7lFTPOMfRFw).
>
> The application is not a consumer product and has no end users other than me. It
> never requests authorization from visitors to the website; no visitor signs in to
> anything. It authorizes against one Google account — my own — and acts only on that
> account's own channel.
>
> **youtube.upload** is required because the application's entire purpose is to
> create videos: it calls `videos.insert` to upload a video file it has just
> assembled, together with the title, description and tags for it. There is no
> narrower scope that permits uploading a video.
>
> **youtube.readonly** is required only to confirm, at setup time, that the
> credential is attached to the intended channel before anything is uploaded. It
> calls `channels.list` with `mine=true` and reads the channel's id and title. This
> guards against uploading to the wrong channel; without it the application cannot
> tell which channel it has been authorized against.
>
> No other Google user data is accessed. Nothing is read from any other user's
> account, nothing is sold or transferred to third parties, nothing is used for
> advertising or for training AI models, and the credential is stored only on my own
> machine. This is stated publicly at https://justcordelia.com/privacy.

If Option B is chosen instead, replace the `youtube.readonly` paragraph with:

> **youtube** is required because the application both uploads its videos and
> maintains them afterwards: it calls `videos.insert` to publish, `channels.list`
> with `mine=true` to confirm the credential is attached to the intended channel, and
> `videos.update` to correct the description of a video it has previously uploaded —
> for example when a clue's explanation is revised, or to re-apply chapter timestamps.
> `youtube.upload` alone does not permit editing an existing video's metadata, and
> `youtube.readonly` alone does not permit the channel check plus the edit. Every one
> of these acts on videos the application itself created, on one channel that I own.

---

## 4. Demo video — shot list

Google requires: the end-to-end flow including the OAuth grant; the same app and
branding as submitted; the **complete** consent screen showing the **exact** scopes
requested; consent screen in English; and a demonstration of the functionality those
scopes are used for.

Record one continuous screen capture, no cuts, no narration needed:

1. **Show the homepage.** `justcordelia.com` in a browser, so the reviewer sees the
   app that the consent screen belongs to. Follow the footer link to `/privacy` and
   scroll to "Use of Google account data".
2. **Start the grant.** In a terminal, run
   `.venv/Scripts/python.exe scripts/youtube_auth.py`. Let it open the browser
   itself — the reviewer should see the flow the app really uses.
3. **Hold on the consent screen.** Do not click through quickly. The app name, the
   account being signed in as, and every scope must be legible and in English. Scroll
   so the full scope list is visible.
4. **Grant it**, and show the terminal reporting the channel it is now attached to —
   that is `channels.list` doing the job `youtube.readonly` is requested for.
5. **Show the functionality.** Run
   `.venv/Scripts/python.exe scripts/youtube_upload.py --build --privacy private`.
   It captures the clue pages, encodes, and uploads. Let the progress percentages run
   — this is `videos.insert`, the reason for `youtube.upload`.
6. **Show the result.** Open the printed watch URL. The video is on the channel,
   private.

Upload it to YouTube as **unlisted** and put the link in the submission. Hosting is
not specified by Google, but unlisted YouTube is the ordinary choice and the channel
already exists.

One caution: step 5 costs 1600 quota units, and a re-record costs another 1600
against 10,000/day. Rehearse the first four steps before recording for real.

---

## 5. What is left for the user

1. Decide §2 — Option A or B. (Recommendation: A.)
2. Check the two console items in §1: app name matches Cordelia's branding, privacy
   policy URL identical to the homepage's.
3. Record §4 and upload it unlisted.
4. Submit with the §3 text.

Steps 1 and 2 are the ones that gate everything else. Nothing in the codebase changes
except the scope list, and only if Option A is chosen.
