# Mapper / verifier improvements — TODO

Captured while walking through clues with the user. Not yet applied.

1. **Link_words sacred principle** — only LINK_WORDS-listed words go into
   `form.link_words`; everything else stays unaccounted. ✓ DONE.

2. **Gloss-first anchoring + shadow candidate when DB lacks the multi-word
   entry** — when the blog supplies a gloss, anchor by phrase-match against
   the clue (not DB lookup). Confirm via DB; if missing, write shadow row.
   ✓ DONE.

3. **Phrase matching: strip punctuation; smallest-span containing content
   words of multi-word gloss.** Currently mapper does exact-phrase
   matching. For glosses like `(German for yes)` the clue text "Yes,
   German" doesn't match because of comma + reorder + extra "for". Fix:
   strip punctuation, ignore stop words (for/of/to/the/a/an), find
   smallest contiguous span of clue words containing every content word
   from the gloss. Anchor to that span. JACKANAPES case.

4. **DD detection path in db_anchored_mapper.** When blog says "Double
   definition" or "DD", split clue into two halves (at apostrophe-s,
   comma, or midpoint), verify both halves are definitions of the answer
   in DB, emit `dd(synonym(left, answer), synonym(right, answer))`.
   Currently dropped when we discarded the old blog_parser. EDEN case.

5. **POS-aware source-span widening when blog under-glosses.** When a
   piece's blog gloss is a single noun (e.g. `(beer)`) but the clue has
   a noun phrase (ADJ+PREP+NOUN, e.g. "associated with beer"), widen the
   piece's source span to the whole noun phrase. POS rule: adjective +
   preposition + noun belong together. The blogger's gloss-on-noun-only
   signals the source span includes the modifiers. Same applies to
   ADV+ADJ+NOUN, NOUN+OF+NOUN, etc. IMPORTER case.

6. **(architectural)** Phase A shadow loop ✓ DONE.

7. **(architectural)** Phase B sacred principle + gloss-first anchoring
   ✓ DONE.

## Open questions / observations

- Compound containers like JACKANAPES (`JA + K both inside CANAPES`) —
  enumerator handles 3-piece container_charade but the 2-pieces-inside-1
  pattern needs verification it works in all cases.
- Anagram fodder when blog has bare uppercase tokens (no `(gloss)`) —
  the anagram-active flag handles `anagram of X Y` but `*(X Y)` and
  variants need cross-checking.
- Encoding/punctuation stripping: `prot g → WARD` showed UTF-8 chars
  getting stripped to garbage. Need to preserve or decode them.

8. **TFTT mixed-case positional notation.** When a token has a single
   uppercase letter and the rest lowercase (e.g. `Bottle`), interpret as
   positional extraction:
   - `Bottle` → first letter of "bottle"
   - `bottlE` → last letter of "bottle"
   - `Bottle` (caps at ends, lowercase middle) → outer letters
   - `bOTTle` → middle/inner letters
   - The bracketed indicator nearby (e.g. `[to begin with]`) gives the
     positional kind. BAIL case.

9. **Floating-op anchoring should skip LINK_WORDS.** When a floating
   op (e.g. container, derived from blog op-word `in`) needs an anchor
   in the clue, currently `_attach_floating_ops` takes the first
   unused leftover. It should skip words that are in LINK_WORDS —
   "of", "in", "with" etc. are connectives, not indicators. IRONMAIDEN
   case: "securing" should get the container indicator, not "of".

10. **Postfix asterisk anagram notation `(fodder)*`.** TFTT uses both
    `*(FODDER)` and `(fodder)*` for anagram. We only handle prefix.
    For postfix: after closing a paren, check for trailing `*`; if
    present, treat the parenthetical content as anagram fodder.
    ELEMENTAL case: `(Lee)*`.

11. **Bracket inheritance from asterisk-anagram.** When a bracketed
    indicator follows a `(...)*` or `*(...)`, the indicator should
    inherit operation type "anagram" so it tags as the anagram
    indicator. ELEMENTAL: `[in a state]` after `(Lee)*`.

12. **Apostrophe-s normalisation.** Words like "that's", "knight's",
    "PM's" should have the trailing 's stripped during tokenisation
    so the base word matches LINK_WORDS / DB lookups. Currently they
    fall through.

13. **Relaxed gloss matching: filter gloss words to those in the clue.**
    Strengthens TODO #3. When the blog's gloss contains descriptive
    words not present in the clue (e.g. "Weasley; Harry Potter's friend"
    when clue has "Harry's friend"), drop the missing words and find the
    smallest span of clue words that covers the remaining gloss words.
    Combine with #12 (apostrophe-s stripping) and basic stem
    normalisation (singular/plural) for word matching. KRONA case.

14. **Stem normalisation (singular/plural) during phrase matching.**
    "friends" should match "friend" and vice versa during gloss/clue
    word matching. Apply during the same matching step as #3/#13.

15. **Bracket inheritance from curly-brace deletion piece.** When a
    piece token contains curly-brace deletion ({s}ETTE{r}, REN{t}, etc.)
    AND a bracket indicator follows, the bracket should inherit the
    "deletion" operation (with the kind already determined by the
    curly-brace position: head/tail/heart/outer). Currently bracket
    inheritance only fires from preceding op-words. ETIQUETTE case:
    `{s}ETTE{r} [uncovered]` — uncovered should tag as the deletion's
    indicator.

16. **Floating-op anchoring: prefer DB-tagged indicator words.**
    Strengthens TODO #9. When a floating op needs an anchor in the
    clue residue, the algorithm should:
    1. Skip leftovers that are in LINK_WORDS.
    2. Among remaining leftovers, query the indicators DB for each
       word — prefer ones tagged with the matching op type
       (container, reversal, anagram, etc.).
    3. If multiple candidates match, fall back to position heuristic
       (closest to the pieces, or in mid-clue).
    4. If no match, take the first non-link leftover.
    AXIS case: leftover {Turning, point, hosting} for a container op;
    "hosting" is in DB as container, "Turning" as reversal — pick
    hosting.

17. **Definition anchored BEFORE pieces.** Currently pieces anchor first,
    definition second. When a piece's gloss matches a clue word inside
    the definition span (e.g. "drink" in "Drink supplier"), the piece
    consumes it and the definition can't anchor cleanly. Fix: when
    `clue.definition` is supplied, anchor it first; pieces work around
    those used indices. WAITER case.

18. **Allow LINK_WORDS-listed words to be indicators when blog
    confirms.** Specifically: `in`, `of`, `with`, `at` can be either
    link or container/positional indicators. When a floating op needs
    an anchor and a link-word-listed clue word matches the op type
    in indicators DB (or aligns with the blog's op-word), let the
    indicator role override. Sacred principle preserved (still no
    silent absorption — the blog is the witness). WAITER case.

19. **(known limit)** Discontinuous multi-word indicators like
    "put...in", "put...inside". Cryptic convention but mechanically
    hard. Out of scope for this round; LLM territory.

20. **`->` derivation arrow handling.** Some blogs include the
    intermediate result: `RUN (hurry) [back] -> NUR + SLING (...)`.
    The `->` shows the transformation; the bare token after it is
    the result already produced by the preceding piece+indicator.
    Skip both `->` and the bare token after it. Or recognise: if
    a bare PIECE follows `->`, it duplicates the preceding piece's
    transformed value, drop it. NURSLING case.

21. **Indicators DB lookup for bracketed phrases — replaces #11, #15.**
    Single unified rule: when a bracketed indicator `[...]` appears,
    query the indicators DB for the bracketed phrase (and individual
    words within). Use the matched op type directly. This subsumes
    TODOs #11 (anagram bracket inheritance) and #15 (deletion bracket
    inheritance). Works for all op types: anagram, container, reversal,
    deletion, positional, hidden, homophone. Avoids needing per-op
    context tracking.

22. **Categorise LINK_WORDS into pure_link vs op_separators.**
    - Pure link: "the", "a", "an", "to", "of" (mostly fluff)
    - Op separators: "with", "and", "by", "before", "after", "next to",
      "then" (signal piece boundaries in a charade)
    - Op indicators: "in", "around", "containing", "inside" (signal a
      specific op — likely container)

    Verifier treats pure_link and op_separators equally for residue
    (both contribute no letters). Mapper uses op_separators as
    structural hints when identifying pieces. Critical for future
    no-blog clue parsing.

23. **(known limit)** Compound deletion: deleting a letter that's
    itself sourced positionally from another clue word. Schema's
    deletion op has kind=tail/head/etc. — it doesn't model
    "delete the last letter of word Y, which happens to be in word X".
    STUN case: delete-T-from-STUNT-where-T-is-last-letter-of-gymnast.
    Schema extension required (deletion's removed child is itself a
    positional leaf). Defer.

24. **"for" / "gives" / "→" as definition indicators in the surface.**
    Useful when `clues.definition` is missing or wrong — the surface
    word "for" before the definition can flag where the definition
    starts. Currently we trust `clues.definition`. Add as a fallback
    extraction rule.

(replacing #23) **Two-operand deletion.** Extend `deletion` schema to
accept an optional `removed` child node (parallel to container's
outer/inner). When `removed` is set, the verifier mechanically deletes
that letter sequence from the source. Same shape as container.
STUN case: `deletion(source=synonym(STUNT), removed=positional[last](Gymnast))`.
Not compound — single deletion op with two operands.

25. **DBE markers (definition by example).** Words like "possibly",
    "perhaps", "say", "for example" flag that a clue word is an example
    of a category, not a literal synonym. E.g. "panama, possibly" → HAT
    means HAT is the category, panama is an example. Currently the
    mapper treats the clue word as the synonym source, which fails the
    DB lookup ("panama → HAT" isn't in synonyms_pairs but "panama" IS
    a kind of HAT). Need: recognise DBE markers, treat the adjacent
    noun as an example-of, look up the category via reverse synonym
    chain (or accept the example-of relationship as a confirmed leaf
    type without strict DB hit).

26. **Hidden-word notation with TFTT mixed-case highlighting.** When
    the blog has `Hidden in croquetteS A LA MInute`, the capitalised
    letters (S A LA MI) mark exactly which letters form the hidden
    answer. Generalise #8 (mixed-case parsing) so that for hidden-word
    contexts, the capitals indicate the hidden span rather than a
    positional extraction. SALAMI case.
