# Plan — telegraph 31275 remaining failures: new engines needed (PLANNING ONLY)

Date 2026-06-26. Re-solved all 32 clues of 31275 through the current cascade (no forced
overrides, auto-signature off). 12 hard FAILs + 8 "pending" partials. This plan covers the
ones that need a NEW BESPOKE ENGINE (a shape the cascade can't assemble), not the ones that
only need DB enrichment. No code written — design only.

Method: hand-decomposed every clue, confirmed the letters, then classified.

---

## THE BIG PATTERN — "container with a BUILT inner" (6 clues, the single largest gap)

The container family (`container`, `container_charade`, `container_deletion`, `reversal_container`,
`anagram_container`) only wraps **plain DB values**. It cannot wrap an inner (or outer) that is
itself a *selection / deletion / alternation / small charade*. Five fails are exactly this:

| clue | answer | decomposition | inner mechanism |
|---|---|---|---|
| 10075752 | RIVEN | RIEN(=FRIEND, "discovered"=remove ends) **around** V | first-letter selection ("valuables primarily") |
| 10075776 | PRISONER | PRIER("peeping Tom") **around** SON | ALTERNATION ("scor**n** at intervals" = s,o,n) |
| 10075780 | SINEW | SEW("stitch") **around** IN | NAMED deletion ("pain" − "Dad"/PA = IN) |
| 10075753 | LIMBURGER | LIMBER("Flexible") **around** URG | curtail ("fancy"=URGE, "for the most part" → URG) |
| 10075763 | ASPIC | AC("Bill"/account) **around** SPI | curtail ("turn"=SPIN, "briefly" → SPI) |
| 10075759 | DREDGE UP | DUP("party at Westminster") **around** (RED+GE) | charade w/ a hollow piece (RED + GE="George vacuously") |

All five reuse primitives we already have (`core.selection` first/alternate, `core.deletion`
ops, the synonym lookup). The NEW part is the **container assembly around a constructed inner**,
answer-driven, gated on a container indicator + the inner's own indicator.

**Proposed:** a small family of bespoke "container-with-built-inner" engines, one per inner
mechanism (so each keeps its own verifier, per the architecture):
1. `container_selection` — inner is a first/last/outer-letter selection (RIVEN).
2. `container_alternation` — inner is an alternation selection (PRISONER).
3. `container_named_deletion` — inner is value−namedRun (SINEW).  *(check: does the existing
   `container_deletion` already do INNER deletion, or only OUTER? WHARFS was outer-deletion — so
   inner-deletion LIMBURGER/SINEW are likely uncovered.)*
4. `container_charade` extension — allow the inner to be a 2-piece charade where one piece is a
   hollow/positional deletion (DREDGE UP). May fold into a `container_of_charade` engine.

Each: gated on a container indicator + ≥1 of the inner indicators; build the outer (plain DB
value) and the inner (via the primitive), insert, require the result == answer EXACTLY; own
verifier; additive. A shared helper can do the wrap+verify; the per-engine part is which inner
primitive it calls (keeps them bespoke but non-duplicative).

---

## STANDALONE NEW SHAPES

### 10075760 STAMINA — reversal + substitution where the REMOVED letter is an extraction (HARD)
"Energy of creatures heading west, needing time for meal finally" = STAMINA.
ANIMALS("creatures") reversed("heading west") = SLAMINA; then **T**("time") replaces the letter
**L**, where L is itself an EXTRACTION — "meal finally" = the LAST letter of "meal". So
S**L**AMINA → S**T**AMINA. THREE stacked mechanisms: reversal + letter-extraction (last-of-meal=L)
+ substitution (T for that L). This is the difficult part the user flagged: the substituted-OUT
letter is not a plain named letter — it is a SELECTION, so the engine must extract L from "meal"
before swapping it. The existing substitution engine swaps two wordplay-table letters; it does
NOT handle a from-letter that is a positional selection. HIGH false-pass risk if unconstrained
("extract any letter, swap for any letter" can hit many answers). Needs the tightest answer-driven
gating: the removed letter must be exactly the indicator-licensed selection of a specific clue
word, AND the inserted letter a DB abbreviation, AND the result == answer exactly. Build LATE,
with an overnight A/B, given the risk.

### 10075778 DIE HARD — anagram whose fodder includes an abbreviation
"Film about hotel I dread being remade" = DIE HARD. anagram("being remade") of **H**("hotel"
abbreviation) + "I dread" = DIEHARD. The plain anagram engine uses RAW letters only, so it can't
pull H from "hotel". Bespoke `anagram_with_abbrev`: anagram fodder = raw words + standard
single-letter DB abbreviations, tightly gated (only DB abbreviations, answer-driven exact) so it
does NOT become an indirect-anagram fabricator. (Watch the false-pass risk carefully here.)

### 10075781 IMPEL — charade + alternation (engine EXISTS, needs an order fix)
"Parking behind this writer's Tesla regularly in drive" = IM("this writer's") + P("Parking") +
EL("Tesla **regularly**" = alternate e,l). This is exactly the `charade_alternation` engine I built
— but it FAILS because that engine only looks for the fodder AFTER the indicator (SANDPIPER:
"oddly ignored **near**"). Here the fodder "Tesla" is BEFORE the indicator "regularly". FIX:
extend `charade_alternation` to accept the fodder word on EITHER side of the alternation
indicator. (Small enhancement to a same-session engine, not a wholly new engine. Also needs
"this writer's"→IM, "Parking"→P in the DB — minor enrichment.)

---

### 10075769 PALATIAL — reversal of a charade whose pieces include a deletion
"Sumptuous large Indian dish, not starter, and drink served up" = PALATIAL.
**reverse**(L["large"] + AITA[RAITA "Indian dish" − R "not starter"] + LAP["drink"]) = PALATIAL.
"served up" reverses the WHOLE charade. So it is `reversal_charade` BUT one piece (AITA) is a
beheaded synonym — the existing reversal-charade reverses plain pieces only. NEW shape:
reversal + charade + deletion. Bespoke `reversal_charade_deletion` (or extend the reversal-charade
to allow one positional-deletion piece), answer-driven (reverse(assembly) == answer exactly),
gated on a reversal + deletion indicator.

---

## EXCLUDED — DB ENRICHMENT ONLY (right engine exists, missing data; NOT in scope)

- 10075770 REMISS — needs the multi-word synonym so MISS covers "way of addressing schoolteacher".
- 10075771 LEGISLATE — charade LEG+IS+LATE; "member"→LEG synonym missing (provisional piece).
- 10075774 THESAURUS — container TAURUS around HES; "Sign"→TAURUS missing.
- 10075766 SCREAMING — double-def (laughing + "Carry On Screaming"); add the by-example def.
- 10075767 NEVER / 10075779 PATINA — hidden-reversed; indicator wording ("saving for retirement"
  / "Raised … apparently") not typed.
- 10075777 SUITED — charade SUIT("clubs, maybe", def-by-example) + ED("extremely enraged"); piece
  data.
- 10075758 GREYS — homophone of "graze" ("eat grass" → graze, "audibly"); homophone engine exists,
  needs the synonym/pronunciation.
- 10075761 SHERIFF — pending container/deletion; data.
- 10075775 TYPEFACES — double definition ("Poor Richard and Georgia, say" = typefaces / "put emojis
  in emails?"); dd engine exists, needs the by-example def recognised.

---

## SUGGESTED BUILD ORDER (when we resume, each additive + A/B-gated)

1. **container-with-built-inner family** — biggest payoff (5 clues, recurring shape across the
   corpus, not just this puzzle). Start with `container_selection` (RIVEN) + `container_alternation`
   (PRISONER) since they reuse the selection primitives we just used for SANDPIPER.
2. **charade_alternation order fix** (IMPEL) — tiny, completes a same-session engine.
3. **container_named_deletion** (SINEW) — clear shape, reuses named-deletion primitive.
4. HIGH-RISK, build LAST with the tightest gating + overnight A/B before trusting:
   - **anagram_with_abbrev** (DIE HARD) — indirect-anagram false-pass risk.
   - **reversal + extraction + substitution** (STAMINA) — the removed letter is a selection;
     unconstrained "extract-and-swap" is a false-pass magnet, so gate hard and A/B.
5. **reversal_charade_deletion** (PALATIAL) — reverse of a charade with a beheaded piece.
   (ASPIC now folds into the container family at step 1; TYPEFACES is DB-only.)

Every engine: bespoke, own verifier, answer-driven (exact), additive (never edits a working
engine), A/B-gated to prove 0 regressions (overnight), and ships with its per-type renderer.

---

# MANUAL-SOLVE MODE — spec (for the unfair / derivative clues we will NOT engine)

A HUMAN authoring tool inside the hand-solver for clues the cascade can't and shouldn't solve
(the "operations on derivatives" clues: STAMINA, DIE HARD, and any one-off past the fairness
line). It is **not a solver**: it derives nothing, verifies nothing, and never runs in the
cascade. It is a dumb recorder of what the human types, fired only on an explicit Commit.

## Workflow
1. Open the hand-solver for the clue.
2. For each wordplay piece: tick the word(s), set role = **synonym** (reuse the existing role),
   and type the piece's **contribution to the answer** — i.e. the letters it ends up as AFTER
   the human applies the operations in their head. (PALATIAL "Indian dish, not starter, served
   up" → you type **ATIA**; not RAITA, not AITA — the final reversed form that appears in the
   answer.)
3. Tag the remaining words with their roles as now: definition, indicator(s), link, filler.
4. **Auto-colour**: the typed pieces tile the answer left-to-right in commit order; each piece
   colours its run of answer tiles in its own palette colour (colour only — no positional drag,
   no nesting logic). PALATIAL: PAL(1-3) · ATIA(4-7) · L(8).
5. **Commit**: assemble a Parse from EXACTLY what was typed/tagged and persist it FROZEN. No
   cascade, no verification. **Uncommit**: clear the manual parse and hand the clue back to the
   cascade.
6. Status stays manual (existing pass / pending / fail / INVALID + freeze).

## Hard constraints (what keeps this on the right side of the line we drew)
- **Per-clue values ONLY — never written to the reference DB.** The typed values are derivatives
  ("Indian dish" → ATIA), not real synonyms; writing them would manufacture exactly the junk we
  have been deleting. Manual mode SUPPRESSES the synonym role's normal `admin_db.add_synonym` /
  DB-enrichment write.
- **No derivation** — the system computes nothing; the human types every value.
- **No validation / no auto-pass** — the human sets the status by hand.
- **Never invoked by the cascade** — it is a separate, explicit human path; a committed clue is
  frozen so a later cascade re-run never overwrites it (same mechanism as the frozen status, but
  for the whole parse).
- **Flagged "manual"** on the clue page so it is never mistaken for a derived solve.

## Why this is NOT the removed auto-builder
The deleted builder DERIVED values itself, SELF-VERIFIED (claimed pass), and ran SILENTLY inside
the resolve path, competing with the cascade. Manual mode does none of those: human-typed,
human-committed, human-statused, cascade-untouched, clearly labelled. One automated solver (the
cascade) stands; this is a hand tool for the exceptions it can't fairly reach.

## Build notes (when back at the server)
- Reuse the HS synonym role for the typed pieces; add a manual-mode flag that suppresses the DB
  write and tiles/colours by commit order.
- Add Commit / Uncommit (persist a frozen manual Parse built from the assignments; the existing
  freeze covers status — extend to freeze the whole parse). Render via the existing source-colour
  path; add a "manual" badge.
- Smallest first test cases: PALATIAL (3 pieces, clean reversal) then STAMINA (the 3-op one).
