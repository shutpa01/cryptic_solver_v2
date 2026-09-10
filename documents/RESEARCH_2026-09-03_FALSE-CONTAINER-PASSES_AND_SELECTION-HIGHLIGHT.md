# RESEARCH — 2026-09-03

Three problems, researched to the point of being ready to implement. **Nothing in
`core/` or `web/` has been changed for items 1 and 2.** The one change already
committed today is `b5ee663b` (`container_deletion`), which is the same fault as
item 1 and was fixed on the user's instruction.

The user's framing is the point: *"we have a solver that we can't trust, which
means I have to check 90 clues per day."* Item 1 is the biggest single source of
that distrust that I can measure.

---

## 1. FALSE CONTAINER PASSES — a container that contains nothing

### The fault

A container clue means one string is **inside** another. Every container engine
enumerates the answer as `inner = answer[p:p+L]`, `outer = answer[:p] + answer[p+L:]`.
For that to be a container, the outer must have letters on **both** sides of the
inner: `p > 0 and p + L < N`.

Eight engines test that. **Seven do not** — they only reject the case where the
inner covers the whole string, so an inner sitting at either END is accepted. An
inner at either end is a **charade**, and the container indicator is then badged
onto a parse where it does no work at all.

The user, on seeing one: *"How on earth can we pass something with an unused
indicator? That's BASIC."*

### Where — verified by reading each file

| Engine | Line | Present test | Should be |
|---|---|---|---|
| `core/container_engine.py` | 64 | `if p == 0 and p + L == N` | `if not (p > 0 and p + L < N)` |
| `core/anagram_container_engine.py` | 88 | `if p == 0 and p + L == N` | `if not (p > 0 and p + L < N)` |
| `core/charade_anagram_container_engine.py` | 130, 172 | `if p == 0 and p + L == M` | `if not (p > 0 and p + L < M)` |
| `core/container_charade_signature_engine.py` | 78-79 | `for p in range(0, L)` / `Li in range(1, L - p + 1)` | `range(1, L)` / `range(1, L - p)` |
| `core/anagram_container_signature_engine.py` | 74, 80 | `for p in range(0, Lv + 1)` / `(0, Lf + 1)` | `range(1, Lv)` / `range(1, Lf)` |
| `core/container_signature_engine.py` | 78 | `for p in range(0, Lo + 1)` | `range(1, Lo)` |
| `core/reversal_container_engine.py` | 56 | `for p in range(0, Lo + 1)` | `range(1, Lo)` |

Already correct, and the model for the fix: `container_acrostic_engine.py:53`,
`container_inner_charade_engine.py:78`, `container_inner_deletion_engine.py:194`,
`container_inner_alternation_engine.py:151`, `container_deletion_selection_engine.py:174`,
`container_outer_charade_engine.py:87`, `reversed_outer_container_engine.py:130`,
and `container_deletion_engine.py:229` as of `b5ee663b`.

### Measurement

Cascade-level A/B, 2,752 telegraph/times/guardian clues, last 30 days, on the
`db_only` wiring (the path the nightly actually runs). Strict copies of all seven
engines were built in a scratchpad and patched in at runtime; **the repo was not
touched**. The harness was validated first — it must reproduce a known false pass
(FLAB) before its numbers count.

```
cascade passes — baseline 1765, strict 1745      (-20)
clues whose verdict or engine changed: 52
```

- **20 clues lose a PASS.** Every one was inspected: **20 of 20 are charades
  wearing a container badge.** In each, no source has another source's letters on
  both sides of it — nothing is inside anything. Examples:

  | Clue | Answer | The "container" | The idle indicator |
  |---|---|---|---|
  | Parisian in bar fight | ENCOUNTER | EN + COUNTER | "Parisian" |
  | Tiny child in wartime camp… | STALAGMITE | STALAG + MITE | "in" |
  | Go through curved opening at 135 degrees | SEARCH | SE + ARCH | "opening" |
  | Thin as Oliver on screen? | REEDY | REED + Y | "screen" |
  | Twitch from fish on slab's top | SHAKE | S + HAKE | "on" |
  | Fish firm behind dock making a comeback | PILCHARD | PILC + HARD | reversal "container" |

- **26 clues keep the pass and move to `container_outer_charade`**, which already
  carries the correct interiority test, so by construction those are true
  containers. This looks like a straight improvement in labelling.
- The rest are fail→fail engine relabels.

### The trade-off, stated plainly

Applying this removes ~20 passes per 30 days (~0.7/day). Every one of them was
lying about the mechanism, so the honest outcome is a FAIL, not a pass. Those
clues then need either a data addition or the correct engine. **This does not lose
a single genuine container.**

### Recommended change

One line per engine, seven engines. Per CLAUDE.md rule 4, **one file at a time,
A/B between each**. The harness is written and validated:
`scratchpad/ab_family.py` (+ `make_strict.py`, `inspect_lost.py`).

Order, cheapest blast radius first: `reversal_container` → `anagram_container_signature`
→ `anagram_container` → `charade_anagram_container` → `container_charade_signature`
→ `container_signature` → `container_engine` (the biggest, ~263 stored passes).

### Why not a generic gate instead

A parse's link table cannot always distinguish a true container from a charade
after the fact: where the outer is split across several sources, a 3-piece charade
and a container are structurally identical, and only the clue's indicator
separates them. So the constraint has to live in the enumeration, in each engine —
soundness by design, not a downstream gate. (`feedback_soundness_by_design_not_gates`,
and the verifier is not to be touched: `feedback_never_touch_verifier`.)

---

## 2. WRONG LETTER HIGHLIGHTED — clue 10089267, Guardian 30102 19d

"Blue Ivy Carter initially supports Beyonce's latest nonsense" = **EROTIC**.
"Beyonce's **latest**" is the LAST e; the card lights the FIRST.

### Cause — proven, not inferred

`core/wfw_render.py:603 _selection_rule(parse)` returns **one rule for the whole
clue** — the first selection indicator it finds in the annotations. This clue has
two:

```
'initially'  note='selection/first indicator'
'latest'     note='selection/last indicator'
_selection_rule(parse) -> 'first'          <- applied to EVERY selection piece
```

So "Beyonce's" is highlighted under rule `first`: `B[e]yonce's`. Under its own
rule it is `Beyonc[e]'s`. The stored data is correct — the piece records
`mechanism=selection` and the assignment records `rule: last`. Only the render
re-derives, and re-derives clue-wide.

This is the **second** time this bug has been fixed. 2026-08-09 fixed it for ACUTE
("athlete's" → last E) by reading the rule from the indicator instead of guessing.
That fix assumed one selection rule per clue, so it breaks the moment a clue has
two.

### Both surfaces have it

- `core/wfw_render.py:603` `_selection_rule` → used at `:696` (admin WFW card)
- `web/wfw_read.py:870` `_sel_rule(indicators)` → used at `:1137` (public
  full-explanation overlay; hand-duplicated by house rule, no core import)

`publisher/` has no copy — checked, no matches.

### Two candidate fixes

1. **Interim, small:** bind the rule to the piece rather than the clue — pick the
   selection indicator that licenses *that* source. Some engines already record
   the binding in the note text (`"last-letter selection indicator (of term)"`),
   and `wfw_hs_assignments` records `rule` per piece for hand/prefill solves.
2. **Sound, and what I would recommend:** stop re-deriving. The selection engine
   knows exactly which letters it took; record them on the piece (the `transform`
   column is already JSON and already used for cuts/reversals — no schema change)
   and have both renderers highlight what is recorded, falling back to the current
   derivation only when absent. This is `piece_transform_recorded_not_derived`
   applied to selections, and it ends the class rather than patching it a third time.

### Not yet measured

I started an audit of how many stored solves are affected and it was interrupted;
it is written (`scratchpad/audit_sel.py`) and takes a few minutes. The shape of the
answer: only clues with **two or more selection indicators naming different rules**
can be wrong, and only where the taken letter repeats in the fodder word.

---

## 3. YOUTUBE — the arbitrary 20-minute timeout

**Already resolved for today**: Guardian 30102 was built but never uploaded. Its
video existed on disk, so it needed no re-filming. Uploaded public:
`https://www.youtube.com/watch?v=Y0OH4sgEWg4` (29 chapters). All three of today's
are now in the ledger.

### Cause

`dashboard/pages/deploy.py:416` — `subprocess.run(..., timeout=1200)`. One
invocation of `youtube_upload.py` films **all three papers in sequence** (its
`--sources` default is `telegraph,times,guardian`). The 20-minute cap is measured
against three full film-and-upload cycles, not one. Guardian is last, so Guardian
is always the casualty. The comment above the call — "20 minutes is generous
cover" — was written thinking of a single puzzle.

Today's timeline fits exactly: telegraph uploaded 06:24:40, times 06:31:42,
guardian's video finished assembling 07:35:50 on a later run — about 21 minutes
in — and the process was killed before its upload.

### Recommended change

Both of:
- run the uploader **once per source**, so a slow Telegraph cannot eat Guardian's
  budget and a timeout can only ever cost one paper;
- give each its own generous timeout (1200s each is then genuinely generous).

Awaiting the user's go-ahead.

---

## 4. YOUTUBE DESCRIPTION LINKS — researched, NOT implemented

The user's question: the videos rank well; can the exact-clue long tail rank too?
And the user's own correction of the original design: *"we thought it would be
better to get people to search for the site so Google would see it, but nobody is,
so I think we are better off with a link."*

### What is true today, measured

Two live searches, 2026-09-03:

- **Puzzle-level query** "telegraph cryptic crossword 31333 every clue explained" →
  our video is the **first result**, above Big Dave's.
- **Exact-clue query** "Adult upset teen's mum for fun" (DT 31333, filmed and
  uploaded 1 Sep) → **neither our video nor our clue page appears.** Big Dave's
  takes it; for the sister clue in the same puzzle, lettersolver.com and
  wordhint.net both rank with same-day exact-clue pages.

So the exact-clue SERP is winnable by sites with no authority — they win it by
being indexed the same day. That is consistent with
[[crawl_collapsed_after_5xx_burst]]: our problem is being seen, not the page.

**Why the video wins one and not the other.** The video's TITLE carries the puzzle
identity, which is what puzzle-level queries match. The exact clue text lives only
in the description/chapters, which is weak matching, and one video is one URL — you
cannot get 30 clue-level landing pages out of it. Chapters produce `&t=` key-moment
entries, but they do not rank independently for long-tail text. **The video cannot
substitute for the clue page in that SERP.** Anyone who says otherwise should be
asked for the query that proves it.

### The current description

`scripts/youtube_upload.py:266 description_for()` writes:

    Still solving? Search justcordelia.com — every clue explained, plus a
    pattern finder, anagram solver and thesaurus built in.

A bare brand mention. **No link at all** — the design the user has now revised.

### What a link can and cannot do — stated honestly

- Links in YouTube descriptions are **nofollow**, and YouTube wraps external ones
  in a `youtube.com/redirect?...&q=` hop. **No ranking equity passes.** Anyone
  promising otherwise is selling something.
- Nofollow is a hint, not a block, and Google does crawl YouTube constantly, so a
  description link is a plausible **discovery** route. It is not a guaranteed one,
  and "Discovered — currently not indexed" is already our largest GSC bucket, so
  the realistic upside is discovery, not ranking.
- **The honest counter-argument, which must not be skipped:** the recent clue
  pages are ALREADY in the sitemap (5,875 URLs) and ALREADY internally linked from
  their puzzle page ([[link_graph_measured]]). So this does not expose hidden URLs.
  The argument for it is narrower: it is a *third* discovery route originating on a
  domain Google crawls hourly, rather than on a site whose Discovery crawl is <1%.
- Independent of SEO, it has a plain human benefit: someone who found the video
  from a puzzle query can reach the clue page in one click instead of being told to
  go and search.

### It fits, but only just — measured over all 45 built videos

The manifest already carries the exact slug for every clue
(`logs/youtube/*/manifest.json`, key `slug`), matching the live route
`web/routes/clue.py:286` `/clue/<slug>`. So per-clue URLs need no new URL shape —
[[feedback_never_propose_url_reduction]] is not engaged.

| variant | over the 5,000 cap |
|---|---|
| bare URL per clue, appended after chapters | **4 of 45** (worst +385 chars) |
| "23 Across — URL" per clue | **19 of 45** |

Largest current description is 2,584 chars, so there is ~2,400 of headroom, and a
30-clue block needs ~2,200-2,900.

### Recommended shape

1. The **puzzle page URL** as a full `https://` link in the first three lines, so
   it is above the "…more" fold where a human will actually see it.
2. The **per-clue URLs after the chapters**, added whole-line only while they fit.
   Chapters must stay complete or YouTube renders no chapters at all, so the link
   block goes last and is the thing that gets trimmed — never the chapters.

⚠️ **Latent defect to fix at the same time:** `description_for` ends with a blunt
`[:4900]`. Nothing hits it today (largest is 2,584), but the moment a link block is
added it will, and a mid-URL or mid-chapter cut is exactly the silent-wrong-output
failure this pipeline keeps producing ([[youtube_pipeline_built]]). Trim by whole
lines, and assert the result is under the cap.

### How we would know it worked

Not by opinion: in Search Console, do the linked clue URLs move out of
"Discovered — currently not indexed", and does URL Inspection start reporting a
referring page at all (it currently reports none)? Give it a few weeks. If nothing
moves, the conclusion is that the discovery route does not fire, and we stop.

---

## Harness lessons banked today

Three ways a harness lied to me before I caught it, all now in memory
(`harness_must_install_role_predicates`):

1. **`role_validity.set_predicates` must be installed** (`engine_registry.py:630`)
   or every parse fails and an A/B reports "0 passes" on both sides — which reads
   as "no regression". This produced the false "0 passes lost" in `6ea7b10e`.
2. **Never pass `clue_id` to `solve_clue_text` in a probe** — it PERSISTS
   (`engine_registry.py:1735`, and again at 1781). One did, and overwrote a live
   solve.
3. **Use `db_only(wiring)`** — the full wiring leaves `define_fallback`,
   `suggest_piece`, `value_check`, `ai_is_definition`, `suggest_hom` live, so a
   corpus sweep makes an AI call per clue. Slow, costly, and not what the nightly runs.

And: **a harness must reproduce a known result before its numbers count.** Every
sweep above asserts that first and aborts if it fails.
