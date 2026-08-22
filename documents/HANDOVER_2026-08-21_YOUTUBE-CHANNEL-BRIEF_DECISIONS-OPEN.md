# HANDOVER 2026-08-21 — YouTube channel brief, decisions open

Status: nothing built, nothing committed. This is a brief only.

## The brief

A daily YouTube video that cycles through the WFW pages of that day's Daily
Telegraph puzzle. Frames are captured headlessly from the live clue pages as
they already render — no new renderer, no second content path. Assembled with
ffmpeg, uploaded automatically via the YouTube Data API.

## SEO substance

Three things carry the weight:

1. **Exact puzzle number in the title.** The searches are exact and time-boxed
   ("telegraph cryptic 31234", "DT 31234 hints"). Volume per puzzle is small,
   intent is total.
2. **Auto-generated per-clue chapters.** One chapter per clue, titled with clue
   number, clue text and enumeration — not the answer. YouTube surfaces chapters
   as separate "key moments" entries in Google video results, so one upload
   becomes ~28 timestamped, independently indexable targets. Generated free from
   the puzzle row.
3. **A fixed daily publish slot**, after the nightly run. Publishing before the
   audience stops looking decides whether the other two matter.

Secondary: one clue as a Short each day, off the same pipeline — that is where
subscribers come from, since the long video only serves people already searching.

## What this will NOT do

YouTube description links are nofollow. No link equity for the site. The return
is referral clicks and a second surface Google indexes independently. Given the
Bing situation that independence has value, but do not budget for a ranking lift
on the site's own pages from this.

## Open decisions — these shape the build

1. Narrated or silent? Watch time is YouTube's real ranking signal.
2. Every clue, or a subset with the site as the payoff? Whole puzzle gives better
   watch time; subset gives better click-through.
3. Show the Telegraph clue text on screen, or answer and breakdown only?
4. DT cryptic only, or Times and Toughie too?
5. Saturday prize and Sunday prize-toughie — same day, or held a week?

## Verified facts (checked 2026-08-21, cite before relying on)

- The WFW page is the clue page: `web/routes/clue.py:286`, route `/clue/<slug>`.
- Puzzle page: `web/routes/puzzle.py:22`, `/<source>/<puzzle_type>/<int:puzzle_number>`.
- Prize puzzles are served publicly today — `web/serving.py:33-34` lists
  `("telegraph", "prize")` and `("telegraph", "prize-toughie")` in `SERVED_BROWSE`.
  There is no date or embargo condition in the serving rule. Decision 5 above is
  therefore a real gate, not a formality.
- Nothing video-related exists anywhere in the repo.

## Caution

Copyright exposure differs by surface. A Telegraph complaint against a web page
is a page; against a channel it is a strike, and three strikes ends the channel.
That is the weight behind decision 3.

## Note on the session that produced this

The first pass invented an architectural problem — a warning about writing a
second WFW renderer, sourced from memory rather than the code, when the user had
already said the existing page gets rendered. It was withdrawn. The next thread
should take the brief above and go straight to the five decisions.
