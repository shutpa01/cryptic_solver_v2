# Daily Instagram / Facebook reel — research, 1 September 2026

Research only. Nothing built, no account created, no API called, no money spent,
nothing posted.

**Status of the sources.** ✅ marks a fact taken from Meta's or ElevenLabs' own
documentation, or measured here. ⚠️ marks one that is still only from developer
blogs — they agree with each other, but they are not primary and **three of them
turned out to be wrong** when checked (§3a). Treat every remaining ⚠️ as
unconfirmed.

---

## 1. The finding that changes the plan

**You cannot put a clickable link on a reel. Not on Instagram, not on Facebook,
not in the caption, not as a sticker.** ⚠️ Instagram renders a URL in a reel
caption as plain text; Facebook blocks clickable links in reel captions
deliberately, to stop click-jacking, and in 2026 also began converting links in
comments on popular posts to plain text to suppress link farming.

The brief is "push people to the website, mainly the website". The reel itself
cannot do that. What can:

| Route | Clickable | Automatable | Notes |
|---|---|---|---|
| Profile bio link | Yes | n/a — set once | The only always-on link. One URL. |
| Story link sticker | Yes | ⚠️ Stories are publishable via the same content API | A story is a second daily post, 24h life |
| **Comment-to-DM auto-reply** | Yes, inside the DM | **Yes, official API** | ⚠️ one automated private reply per comment, within 7 days; 750 calls/hour |
| Caption "link in bio" | No | — | Reported CTR 1–2% unoptimised, 10–15% optimised |

**The comment-to-DM route is the only automatable path from a reel to a URL,
and it is a supported Meta feature, not a hack.** The pattern: the reel says
"comment WHY and I'll send you the full explanation", the automation DMs the
clue-page link. It also manufactures comments, which feeds the ranking signals
in §2.

This is a design decision for you, not a technical one, and it has a cost: it
asks the viewer to do something before they get anything. The alternative —
bio link only — asks nothing and converts far less. I have not tested either
and neither has anyone else here.

---

## 2. What actually drives reach, and what does not

⚠️ Sourced from summaries of Adam Mosseri's public statements, consistent
across several 2026 write-ups.

**The three signals named publicly: watch time, likes per reach, and sends (DM
shares).** Sends are reported as the most heavily weighted for distribution — a
private share reads as high-trust. Reels favour watch time and sends; Explore
favours engagement velocity.

Consequences for our format:

- **Completion percentage beats absolute length.** A 20-second reel watched to
  70% outranks a 60-second one watched to 30%. Instagram permits 3 minutes but
  reach to non-followers reportedly favours under 90 seconds.
- **Burned-in captions raise watch time**, because most viewing is silent. This
  matters more than the voice for the ranking signal, though not for the persona.
- **Hashtags are capped at five and, in Mosseri's words, do not improve reach.**
  One study cited shows hashtag-heavy posts reaching 23% *less*. Use three to
  five for categorisation and stop thinking of them as a lever.
- **The 90-second API ceiling matches the algorithmic sweet spot anyway** (§3),
  so the constraint costs us nothing.

**What this says about the brief.** "Say all the puzzles that have been
uploaded" is a listing, and a listing is the weakest possible opening for a
watch-time-driven format — it is information for someone already looking for us,
shown to a feed of people who are not. Worse, it is the exact thing you
yourself struck out on 2026-08-29: *"Nobody scrolling a feed knows or cares
what [publication and number] is."*

I am not dropping it — it is your instruction and it serves a real purpose, which
is proving daily freshness and completeness. But it should not open the reel. The
shape that satisfies both:

1. **0–3s** the clue, on screen, no branding, no logo, no roll-call.
2. **3–35s** Cordelia solving it with the tools, first person.
3. **35–45s** the answer and the mechanism.
4. **45–55s** *"I've explained every clue in today's Telegraph, Times and
   Guardian — that's 93 clues, all on the site."* The roll-call, as a payoff,
   once the viewer has a reason to care.
5. Caption carries the full list in text, where it costs no watch time and is
   searchable.

The roll-call is the *proof*, not the hook. That ordering is a recommendation,
not a decision I have made for you.

---

## 3. Publishing: what Meta requires

⚠️ All of this needs confirming against Meta's docs before code.

### 3a. Where the blogs were wrong

Checked against Meta's own docs, three widely-repeated claims are false. They
matter, because two of them would have caused real work to be done for nothing.

| Blog claim | ✅ Meta's documentation |
|---|---|
| "25 published posts per 24 hours" | **100** per rolling 24h on Instagram; **30** on Facebook Reels |
| "Business account only, creator accounts unsupported" | "**Professional** Instagram accounts connected to a Facebook Page" — professional covers both |
| "the video must be at a public `video_url`" | A **resumable upload** endpoint exists: `rupload.facebook.com/ig-api-upload/<container-id>`. A public URL is one option, not the requirement |

**The third correction removes a problem I had flagged as significant.** I wrote
that the mp4 must be publicly fetchable and that Cloudflare-only mode would
fight us. It need not be: we can upload the bytes directly on both platforms and
never expose the file at all. Nothing needs to change about the Cloudflare
posture.

### Instagram ✅

- **Account**: a *professional* Instagram account connected to a Facebook Page.
  Accounts subject to Page Publishing Authorization must complete it first.
- **Permissions**, and there are two different routes:
  - Instagram Login: `instagram_business_basic`,
    `instagram_business_content_publish`
  - Facebook Login: `instagram_basic`, `instagram_content_publish`,
    `pages_read_engagement`, plus `ads_management` / `ads_read` if the user
    holds the Page role through Business Manager
- **Publish flow**: `POST /<IG_ID>/media` with `media_type=REELS` (with either
  `video_url` or a resumable upload to
  `https://rupload.facebook.com/ig-api-upload/<container-id>`), poll
  `GET /<IG_CONTAINER_ID>?fields=status_code`, then
  `POST /<IG_ID>/media_publish` with the `creation_id`.
- **Rate limit**: 100 API-published posts per rolling 24 hours. Current usage is
  readable at `GET /<IG_ID>/content_publishing_limit`.
- ⚠️ **App review**: reported 2–4 weeks, a separate submission per permission
  with a screencast. Not confirmed from primary sources, and it is the long pole
  — nothing can be tested end to end until it clears.
- ❌ **Reel specifications are not in the pages I could reach.** Build to
  Facebook's published spec below; it is stricter and near-certainly a safe
  subset.

### Facebook ✅

- Reels post **only to Pages** — not profiles, not groups.
- **Permissions**: `pages_show_list`, `pages_read_engagement`,
  `pages_manage_posts`, and a Page access token with `CREATE_CONTENT`.
- **Flow**: `POST /<page-id>/video_reels` with `upload_phase=start`, upload to
  `rupload.facebook.com/video-upload/<video-id>`, then `video_reels` again with
  `upload_phase=finish`. Interrupted uploads resume from `bytes_transfered`.
- **Rate limit**: 30 API-published posts per rolling 24 hours.
- **Published spec — build to this:**

| | |
|---|---|
| Container | .mp4 recommended |
| Frame | 9:16, 1080×1920 (minimum 540×960) |
| Duration | **3 to 90 seconds** |
| Frame rate | 24–60 fps, fixed, progressive |
| Video | H.264, H.265, VP9 or AV1; 4:2:0 chroma; closed GOP |
| Audio | AAC, stereo, 48 kHz, 128 kbps or better |

- ⚠️ **A trap if we ever use the hosted-file route**: "the API will reject files
  hosted on sites that restrict access via robots.txt." **Checked — we are
  fine.** `web/routes/seo.py:58` serves `User-agent: * / Allow: /` with only
  `/admin/`, `/reveal` and `/explain` disallowed. Just do not serve the mp4 from
  under those three paths.

**Two code paths, not one.** Instagram creates a container then publishes;
Facebook runs a three-phase upload session. Sharing anything but the encoder
between them is false economy.

### Is automated daily posting allowed?

⚠️ Nothing found suggests scheduled or automated publishing through the official
API breaches policy — the content publishing API exists for exactly this, and
the comment-to-DM products are built on Meta-approved interfaces. The risk is
not automation, it is *repetition*: near-identical daily reels are the shape
platforms suppress. Mitigation is genuine variation in the clue, the tools shown
and the opening line, which we get for free because the clue is different daily.

---

## 4. AI disclosure — where we stand

⚠️ Meta's 2026 position, as reported: disclosure is **enforced for advertising**
via a control in Ads Manager, and lighter for organic. The categories that
attract a label include **synthetic voice**, photorealistic AI images and
AI-manipulated realistic media. Light-touch AI editing does not.

We use a synthetic voice, so:

- **If we ever pay to promote a reel, we must set the AI disclosure.** Not
  optional.
- For organic posts the requirement is softer, and Meta applies its most visible
  label to photorealistic AI *presenters*, which we are not — Cordelia is not a
  face, she is a voice over a screen recording.

This does not reopen the disclosure question you settled on 08-30. You decided
Cordelia is a machine and has never claimed otherwise, and that stands. This is
about Meta's own labelling controls, not about our honesty.

---

## 5. The voice

- Voice: **Emmaline — "young British girl"**, in the ElevenLabs voice library.
  ⚠️ A third-party voice directory lists her ID as `nDJIICjR9zfJExIFeSCN`.
  **Verify this in your own ElevenLabs account before wiring it in** — a wrong
  ID silently produces the wrong voice, and every reel would carry it.
- ✅ **The call**: `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`.
  Only `text` is required. `model_id` defaults to `eleven_multilingual_v2`.
- ✅ **`voice_settings` is where the persona is tuned**: `stability` (0.5),
  `similarity_boost` (0.75), `style` (0), `speed` (1), `use_speaker_boost`.
  These matter more than the voice choice — a "young British girl" preset will
  read breathless by default, which is the opposite of the unhurried, no-ego
  Cordelia. Settle them once by ear and freeze them, or the channel drifts in
  tone from day to day.
- ✅ **Output format**: mp3 from `mp3_22050_32` up to `mp3_44100_192`, plus Opus,
  PCM, WAV and μ-law. **192 kbps mp3 requires the Creator tier or above.** For
  reels, PCM or WAV into ffmpeg avoids a needless mp3 generation before the AAC
  encode Facebook specifies.
- ⚠️ **Commercial rights require a paid plan.** API output is commercially
  licensed; the free tier is not. Confirm your plan — "narrated by Emmaline,
  posted to a business account, driving traffic to a site" is unambiguously
  commercial. Per-character pricing was not stated on the API page I read.

**The script problem, restated because it has not gone away.** Our stored
explanations are written to be *read*: brackets, `abbreviation=` notation, bare
letter strings. None of that survives being spoken. A text-to-speech step sits
between the explanation and the audio and it is real work, not a filter — it has
to turn "CAD (synonym=rotter) inside CREDIT" into a sentence a person would say.

---

## 6. What we already have, and what is missing

**Reusable:**
- The capture path renders the real card in-process from the same templates.
  Do **not** repoint it at the live site — Cloudflare challenges headless Chrome.
- ffmpeg assembly exists, with its traps documented: `-shortest` is banned, use
  `-t`.
- ~90 explained clues every morning from the DT, Times and Guardian, as a
  by-product of a run that happens anyway. This is the asset nobody else has.

**Missing, and these are the actual build:**
1. **Motion.** Everything the YouTube pipeline produces is stills cut together.
   A tool *in use* — the pattern finder narrowing, a hint opening, letters being
   typed — is live screen recording, which has never been built. This is the
   main piece of work.
2. **1080×1920 — less missing than the 08-30 note implies.**
   `scripts/youtube_short.py` is 193 lines and already builds a 1080×1920 short
   from two captures of the real clue page — the card hidden (clue only, a
   question), then the card shown (answer and wordplay, the payoff) — via the
   same selenium capture and ffmpeg assembly the long video uses. It deliberately
   does not upload. **It has still never been run**, so it is untested rather
   than absent, and the tease/reveal structure it encodes is a good starting
   point that conflicts with one thing: 08-30 settled that Cordelia names the
   answer early and spends the reel on the mechanism, because withholding is the
   sermon. The script's two-frame shape is the withholding shape. Reuse the
   machinery, not the structure.
3. **Burned-in captions**, which §2 says matter more than we assumed.
4. **The speech script step** (§5).
5. **Two publishing clients** (§3), plus the comment-to-DM responder if you take
   that route.

---

## 7. The automation ceiling — the one real conflict in the brief

You said **"it must be automated"**. The 08-30 plan records that Cordelia says
so when a clue is silly or unfair, and that **fairness is not machine-derivable**
— the solver has confidence scores, not taste. Every such call is a human
editorial judgement.

These cannot both hold. Three ways out, in my order of preference:

1. **Automate the daily reel; drop the fairness verdict from it.** The daily reel
   is the mechanism of one clue, fully automatable. Fairness commentary moves to
   occasional hand-made posts. Keeps the persona, keeps the automation, at the
   cost of the daily reel being slightly less opinionated.
2. **Automate to a draft, you approve in one tap.** A reel a day appears in a
   queue; you approve or skip. Not automated by your definition, and it makes the
   channel depend on you being there daily.
3. **Derive a weak fairness proxy** — clue types our solver fails, unusually rare
   indicators, very long wordplay chains. ⚠️ This is exactly the kind of
   "everything else" classification you have banned, and I would not build it
   without you explicitly overruling that.

**Clue selection can be fully automated** against the principle already settled:
the one most people would get from cross letters without ever seeing the
wordplay — gettable answer, invisible mechanism. That is expressible from data we
hold (answer commonness against wordplay complexity), and it needs no taste.

---

## 8. What I would do first, if it were mine

In order, and the first item is the only one with a long lead time:

1. **Start the Meta app review now.** 2–4 weeks, blocks everything, costs nothing
   to begin. Business Instagram account linked to a Page, then submit both
   permission requests.
2. **Test the premise before building the pipeline.** Your own 08-30 note says
   nothing has established that a clue from today's paper stops the scroll, and
   that it is cheap to test by posting. Three hand-made reels would answer it. A
   pipeline built before that answer is a pipeline built on a guess.
3. **Solve the motion problem**, because it is the long build and everything
   visual depends on it.
4. Then the script step, the captions, the publishing clients.

---

## 9. Open questions I could not answer

- **Instagram's own reel specifications.** Not present in the reference pages I
  could reach; Facebook's spec is the working substitute.
- **App review duration and what the screencast must show.** Only blog-sourced.
  This is the schedule risk for the whole project.
- **The Emmaline voice ID**, against your own ElevenLabs account.
- **Whether your ElevenLabs plan carries commercial rights**, and its
  per-character cost.
- **Whether beginners are on Instagram following crossword accounts at all.**
  Still unknown, still unmeasured, and the entire channel rests on it. The 300k
  figure that started this came from an enthusiast account; nobody has
  established what share of that audience is new to cryptics.
- **Whether a clue from this morning's paper stops the scroll.** Your own 08-30
  note, unchanged: cheap to test by posting, and a pipeline built before that
  answer is a pipeline built on a guess.

*Resolved since the first draft of this document: the Cloudflare / public-URL
problem, which does not exist — see §3a.*

## Sources

Instagram/Meta API and policy: [Phyllo — Reels API guide](https://www.getphyllo.com/post/a-complete-guide-to-the-instagram-reels-api) ·
[Phyllo — Instagram API integration](https://www.getphyllo.com/post/instagram-api-integration-101-for-developers-of-the-creator-economy) ·
[Postproxy — Reels publishing](https://postproxy.dev/blog/instagram-reels-api-publishing-guide/) ·
[Netrows — Graph API 2026](https://www.netrows.com/blog/instagram-graph-api-guide-2026) ·
[Meta — Publish a Reel](https://developers.facebook.com/docs/video-api/guides/reels-publishing/) ·
[Ayrshare — Facebook Reels API](https://www.ayrshare.com/blog/facebook-reels-api-how-to-post-fb-reels-using-a-social-media-api/)

Reach and ranking: [Hootsuite](https://blog.hootsuite.com/instagram-algorithm/) ·
[Sprout Social](https://sproutsocial.com/insights/instagram-algorithm/) ·
[TrueFuture Media](https://www.truefuturemedia.com/articles/instagram-reels-reach-2026-business-growth-guide) ·
[Creatorflow](https://creatorflow.so/blog/instagram-algorithm-2026/)

Links and DM automation: [Paperbell](https://paperbell.com/blog/how-to-add-a-link-to-instagram-posts/) ·
[Sprout — link in bio](https://sproutsocial.com/insights/link-in-bio/) ·
[Postproxy — comment to DM](https://postproxy.dev/how-to/instagram-comment-to-dm-private-reply/) ·
[Inrō](https://www.inro.social/blog/instagram-comment-to-dm-automation) ·
[Social Media Examiner — Facebook link rules](https://www.socialmediaexaminer.com/what-facebooks-new-link-rules-mean-for-your-2026-strategy/)

AI disclosure: [Cinerads](https://www.cinerads.com/blog/ai-ad-disclosure-requirements) ·
[AuditSocials](https://www.auditsocials.com/blog/meta-ai-generated-content-label-policy-2026)

Voice: [ElevenLabs voice library](https://elevenlabs.io/voice-library/social-media) ·
[json2video voice directory](https://json2video.com/ai-voices/elevenlabs/voices/nDJIICjR9zfJExIFeSCN/) ·
[BIGVU — ElevenLabs pricing 2026](https://bigvu.tv/blog/elevenlabs-pricing-2026-plans-credits-commercial-rights-api-costs/)
