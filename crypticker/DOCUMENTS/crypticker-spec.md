# CRYPTICKER — Functional Specification v1.0
**Date:** 21 July 2026
**Owner:** Paul
**Reference implementation:** `crypticker-prototype.html` (attached — the working v2 prototype; treat its interaction design, layout and visual language as the source of truth for the front end unless this spec contradicts it)
**Data source:** `C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db` (read-only at serve time, per house rules)

---

## 1. Concept

A daily, timed cryptic-clue assembly game. The player is shown a sequence of real, verified cryptic clues from the archive. For each clue, the wordplay pieces ("atoms") of the answer are displayed as colour-coded tiles **in clue-word order** (which is rarely answer order). The player taps tiles to assemble the answer in the correct order.

- **Fixed time: 2 minutes per round.** Score = number of clues correctly assembled.
- **Same puzzle sequence for every player on a given day** (fair comparison, shareable score).
- One round per day per player.
- Content is drawn exclusively from **old clues** (archive material, not current or recent puzzles) to avoid market overlap with publishers.
- Design principle throughout: **the game must need almost no explanation.** Difficulty comes from content and presentation dials, never from new rules.

---

## 2. Player-facing rules (complete)

1. Tap the pieces to assemble each answer. Tap a placed piece to take it back.
2. Colours tell you what each piece is (synonym, abbreviation, container, anagram fodder). Pieces appear in clue order — the answer usually doesn't.
3. Easier clues: the letters inside each piece are jumbled, but the definition is highlighted in the clue. Harder clues: clean pieces, no definition marked.
4. Skip costs 30 seconds.
5. One ✦ Bonus per round: press it to gamble. You are asked a surprise question about the current clue (you don't know the question until you press). Answer correctly and the clock freezes until you finish that clue. Answer wrongly and the clock keeps running. Either way the Bonus is spent.
6. Two minutes. Same puzzles for everyone. Score = answers built.

Nothing else. Any proposed feature that requires adding a rule to this list needs sign-off.

---

## 3. Game mechanics (implementation detail)

### 3.1 Assembly
- Tiles render in **clue-word order** (the order in which each atom's source word appears in the clue surface).
- Tapping a tray tile appends it to the answer rail; tapping a rail tile returns it to the tray.
- When the number of placed tiles equals the atom count, auto-check: concatenation of the placed atoms' **true text** must equal the answer exactly.
  - Correct → green state, tiles flip to true (unjumbled) letters if the clue was in jumbled mode, brief pause (~650 ms), auto-advance.
  - Incorrect → shake animation, tiles stay placed and editable. No penalty beyond lost time.
- Duplicate letters/atoms are interchangeable: correctness is judged on the concatenated string, not on tile provenance (e.g. TRIPTYCH's two T tiles are equivalent regardless of which fodder word each came from).

### 3.2 Difficulty modes (per clue, not per round)
Difficulty is a per-clue recipe of independent, display-level dials:

| Mode | Atom letters | Definition | Typical use |
|---|---|---|---|
| `easy` | Jumbled within each atom (e.g. HAT → THA) | Highlighted in the clue text | Round openers |
| `hard` | True order | Not marked | Mid/late round |
| `anagram` | Single-letter tiles, in fodder order, shaded by source fodder word | Not marked (configurable) | Pace change, usually final slot |

- Jumbles are **precomputed and stored** per atom (deterministic; same for all players; must not accidentally equal the true order; single-letter and two-letter atoms are never jumbled — two-letter atoms in easy mode are served true).
- Additional dials available to the curator without rule changes: atom count, device type, answer length, clue surface length.

### 3.3 Timer
- 120 seconds, counting down; visible as a shrinking bar plus mm:ss.
- Under 15 seconds: bar and readout turn red (accent `#d64545`).
- At 0: round ends immediately, current clue marked "not reached" unless already solved.

### 3.4 Skip
- Skip button always available, labelled with its price ("Skip −0:30").
- Deducts 30 seconds (constant `SKIP_PENALTY`, tunable). If that takes the clock ≤ 0, the round ends.
- Purpose: prevents scanning ahead to cherry-pick; makes skip a genuine surrender of ~¼ of the round.

### 3.5 Bonus gamble
- One per round. Button disabled after use. Share line appends ✦ if used.
- On press: **clock freezes immediately** (deliberation over the question is never punished), modal shows one multiple-choice question (3 options) about the *current* clue.
- Question is unknown to the player until pressed. Question types, all derivable from the WFW parse:
  - "What is the definition?" (options: definition phrase + 2 other clue fragments)
  - "Which word is the [anagram / container / insertion / reversal] indicator?"
  - "Which clue word gives [ATOM]?" (options: true source word + 2 other clue words)
- Correct → clock stays frozen until the player solves or skips this clue ("✦ CLOCK FROZEN" flag visible). Wrong → clock resumes after ~1 s verdict display; bonus spent; no further penalty.
- Question selection is deterministic per clue (all players who press Bonus on clue N get the same question).

### 3.6 End of round
- Results screen: big score (N of 6), spoiler-free share text, "How they worked" reveal.
- **Share text format:** `CRYPTICKER №{puzzle_no} · {score}/6{ ✦ if bonus used}` + newline + one square per clue (🟩 solved, ⬜ skipped/not reached). Copy-to-clipboard button. Must never leak answers, atoms or roles.
- **Reveal (the education payload):** for every clue — solved or not — show clue text, the atoms as colour chips in correct assembly order → answer, and a one-line prose explanation (source-attributed: publication, puzzle number, position). This is where the player learns cryptics; do not skimp on it.

---

## 4. Content pipeline (the real work)

### 4.1 Source and constraints
- All clues come from `clues_master.db` WFW-verified parses (PASS status only). The verifier's guarantee — every word accounted for, every piece DB-licensed — is what makes every puzzle unambiguous. **Never serve a clue that did not pass verification.**
- **Archive-only rule:** exclude clues newer than a cutoff (proposed: ≥ 5 years old; Paul to confirm). Store publication date with each clue and enforce in the selection query.
- The DB is **read-only** to this system. The pipeline may cache/export, never write.

### 4.2 Role mapping (WFW → player-facing tile colours)
Initial mapping — Paul to review and amend:

| WFW element | Tile bucket | Colour |
|---|---|---|
| Synonym (incl. example/instance substitutions) | `syn` | amber `#e8a33d` |
| Abbreviation | `abbr` | blue `#4a9fd8` |
| Container shell pieces (the outside, incl. split shells) | `shell` | violet `#8b6fc7` |
| Container contents | `syn` (coloured by what the content *is*) | as above |
| Anagram fodder, word 1 / word 2 / word 3… | `anag`, `anag2`, … | teal `#2fa48e`, olive `#8faf3e`, +1 more if needed |
| Literal/carried text | TBD — propose `syn` for v1 | — |
| Definition (clue highlight, not a tile) | `def` | blue underline `#2f6fed`, per hand-solver style |

Player-facing legend words: *synonym, abbreviation, container, anagram, definition*. No device jargon anywhere in the game surface.

### 4.3 Clue selection criteria (launch pool)
A clue is eligible when ALL of:
1. Verified PASS; publication date before the archive cutoff.
2. Device structure in the launch whitelist: charade, container, charade+container combinations, anagram (with ≥ 2 fodder words for the multi-shade effect). **Excluded at launch:** double definitions, cryptic definitions, &lits, reversals, deletions, selections, homophones — anything where assembly order is not the whole story. (DD/CD may join later as deliberate single-colour "gotcha" days; that is a curation decision, not a code path, when it happens.)
3. Atom count 2–5 (anagram clues: answer length 7–10).
4. No leakage: no atom's true text appears verbatim in the clue surface (anagram fodder exempt — fodder is definitionally in the clue); definition must not share a salient stem with the answer.
5. Answer is a single word, 4–10 letters, no proper-noun obscurities (curator judgement).

### 4.4 Daily puzzle construction
- 6 clues per day, in a fixed difficulty ramp (proposed default, tunable): 2 × easy (2 atoms), 3 × hard (3–5 atoms, at least one container), 1 × anagram.
- A generation script selects eligible clues, assigns modes, precomputes jumbles and the bonus question, and emits **one JSON file per day**. Human review step before publish (a curation CLI or simple review page that renders the day's six exactly as the player will see them).
- No clue is ever reused (track served clue_ids).

### 4.5 Daily puzzle JSON schema
```json
{
  "puzzle_no": 1,
  "date": "2026-09-01",
  "time_limit_s": 120,
  "skip_penalty_s": 30,
  "clues": [
    {
      "clue_id": 10080626,
      "source": {"publication": "Guardian", "puzzle": "30064", "position": "21A", "pub_date": "2020-01-15"},
      "clue": "Panels damaged \u2013 try pitch",
      "enum": "(8)",
      "answer": "TRIPTYCH",
      "mode": "anagram",
      "definition": {"text": "Panels", "highlighted": false},
      "atoms": [
        {"true": "T", "display": "T", "role": "anag", "source_word": "try"},
        {"true": "R", "display": "R", "role": "anag", "source_word": "try"}
      ],
      "assembly_order": [0, 1],
      "bonus": {"question": "Which word signals the anagram?", "options": ["Panels", "damaged", "pitch"], "correct_index": 1},
      "explanation": "Anagram of TRY PITCH (signalled by \u2018damaged\u2019). Definition: Panels."
    }
  ]
}
```
Notes: `atoms` are in clue-word order; `display` differs from `true` only in easy mode (jumble); `assembly_order` indexes into `atoms`; the front end must contain **no solving logic** — it renders and checks against this file only. (Answer/atoms being client-visible is accepted for v1, as with Wordle; do not add server-side answer checking yet.)

---

## 5. Architecture

- **Front end:** single-page vanilla HTML/CSS/JS, evolved from `crypticker-prototype.html`. Mobile-first, max-width 480 px, no framework, no build step. Respect `prefers-reduced-motion`; tiles keyboard-focusable.
- **Back end:** Flask app (consistent with house stack). Endpoints:
  - `GET /api/puzzle/today` → today's JSON (cacheable; date resolved server-side, Europe/London).
  - `GET /` → the game page.
- **Generation:** offline Python script(s) in the same repo: `select_clues.py` (query + eligibility filter), `build_day.py` (modes, jumbles, bonus, JSON emit), `review.py` or a `/preview/<date>` route (curator check). Scheduled via Windows Task Scheduler initially, consistent with existing projects.
- **Client state:** localStorage for {last played date, streak, best score, share text}. One round per day enforced client-side only in v1 (accept casual replays via incognito, as Wordle did).
- **No accounts, no server-side leaderboard in v1.** Community best-times/leaderboard is Phase 2; design the JSON and results screen so a score submission can be added without rework.

---

## 6. Acceptance criteria (v1 done means)

1. A generation run against `clues_master.db` produces ≥ 30 days of valid daily JSON files with zero manual editing (curator review = approve/swap only).
2. Every served clue is archive-aged, PASS-verified, whitelist-device, leak-checked (4.3 rules enforced in code, not by eye).
3. The full round is playable on a phone: intro → 2-minute round with skip and bonus → results with share text and reveal, matching the mechanics in §3 exactly.
4. Share text copies correctly and never reveals puzzle content.
5. The front end contains no clue data at build time — everything arrives via the daily JSON.
6. Jumbles are deterministic (same day = same jumble for all), never equal to true order, never applied to 1–2 letter atoms.
7. Bonus flow: clock provably frozen during the question; frozen-until-advance on success; resume on failure; single use per round.
8. Reveal screen shows correct parse chips and attribution for all six clues regardless of player outcome.

---

## 7. Open decisions (Paul)

| # | Question | Current placeholder |
|---|---|---|
| 1 | Archive cutoff age for "old clues" | ≥ 5 years |
| 2 | Skip penalty size | 30 s |
| 3 | Wrong bonus answer: any penalty beyond losing the gamble? | No extra penalty |
| 4 | Frozen clock: until clue advance (current) vs fixed duration | Until advance |
| 5 | Role-mapping edge cases: literals; container contents colouring | Literals → syn; contents coloured by own type |
| 6 | Definition highlight on anagram clues | Off |
| 7 | Round composition ramp (2 easy / 3 hard / 1 anagram) | As stated |
| 8 | Name/branding: Crypticker vs alternative; relationship to justcordelia (subdomain? standalone?) | TBD |
| 9 | DD/CD "gotcha" days — Phase 2? | Excluded from v1 |
| 10 | Ads/monetisation placement | None in v1 |

---

## 8. Explicit non-goals for v1
Accounts and login; server-side leaderboards; hard/blind mode (no answer shown); licensing/white-label packaging; native apps; serving current puzzles in any form; any new player-facing rule beyond §2.
