# Nightly PASS REVIEW — read last night's engine passes and hold back the indefensible

An engine pass is a machine's claim that a derivation works. Every mechanical test
it can be given, it has already passed: all clue words accounted for, every
indicator bound to a mechanism, the letters adding up. What none of that catches is
a claim no reader would accept.

On 2026-09-10, STET (Times 29644 1a) was served as a pass on `is -> PET` — a
synonym harvested off a blog. Mechanically flawless, semantically absurd. A week's
audit found two more of the same shape in 186 engine passes: `spas -> RESORTS`
(deriving the answer from the answer) and `arrogant -> AT`.

Your job is to read the claims and say which ones a solver would reject.

## Scope

Today's engine passes only. Get them with:

    python scripts/audit_engine_passes.py --days 1

That already excludes `manual` (the user's own filings) and `prefill` (pending by
construction). Do not widen it. Older clues may be published and are not yours.

## What to judge

For each pass, two questions:

1. **Does every `say X -> Y` hold?** Would a setter accept X as a way of writing Y?
2. **Does every indicator do the job it is badged with?** A container indicator on
   a parse with no containment is doing no work; a deletion indicator that deletes
   nothing is decoration.

And one structural check: **does the wordplay derive the answer from something
other than the answer?** `spas -> RESORTS`, then RE removed, then RE added back, is
circular — it explains nothing.

## THIS NEEDS CRYPTIC KNOWLEDGE, NOT JUST ENGLISH

`on -> LEG` looks exactly as wrong as `is -> pet`. It is correct: in cricket the on
side is the leg side, and it is standard crossword currency. So are `books -> NT`,
`the French -> LE`, `soldiers -> OR`, `setter's -> ME`, `love -> O`.

If you do not know a convention, that is not evidence against it. Leave it.

## The only action available

    from core.pass_review import demote_to_pending
    demote_to_pending(conn, clue_id, reason)

It moves `pass -> pending` and records your reason where the user reads comments.
It refuses everything else — any other status, manual/prefill solves, and any clue
not published today. Those refusals are the safety case: your worst possible
mistake is giving the user something extra to look at.

**Demote only when you can name the exact claim that fails**, and put that claim in
the reason: "is -> PET: no sense of 'is' means a pet". A vague misgiving is not a
reason. If you cannot write the sentence, leave the clue alone.

## Hard rules

- `demote_to_pending` is your ONLY write. Never call `store.set_status`,
  `store.save_parse` or `set_frozen`. Never write the reference DB — not to delete
  the row that caused a bad pass, however obvious it is. Report it; the user decides.
- Never promote anything. There is no path from fail or pending to pass, and you
  must not build one.
- Working scripts go in a temp directory or `logs/`, never the repo root.
- No server restarts. No re-running clues for score.
- Finish by writing `logs/pass_review_YYYY-MM-DD.md`: how many passes read, how many
  held back and why (one line each), and — separately — any reference-DB rows you
  believe are wrong, with the clue that exposed them. That list is the durable
  value: it turns one bad night into a permanent fix.
