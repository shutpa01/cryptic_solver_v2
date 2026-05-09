# Mapper rebuild — DB-anchored design sketch

## Why rebuild

The current mapper (`word_role_mapper.py`) anchors pieces to clue words
only when the blog provides an explicit `(gloss)`. TFTT's terse format
omits glosses, expecting the reader to use the clue alongside the blog.
The rebuilt mapper uses the **DB** as the anchor source instead of
gloss text, which handles both formats.

## Inputs

```
clue_text      : str        — original clue
answer         : str        — known answer
blog_text      : str        — TFTT explanation
clue_definition: str | None — the value of `clues.definition` (when set)
db             : RefDB      — read-only access to cryptic_new.db
```

The `clue_definition` column is populated for many clues by prior
extraction. We trust it as a hint, with fallback by-elimination if
absent or contradicted.

## Output

Same `Mapping` dataclass as today, plus a new `shadow_candidates` list
holding (kind, source_word, value) triples that aren't in the live DB
and could populate the shadow store.

## Algorithm

### Stage 1 — Tokenise blog into events

Walk the blog left-to-right producing an event stream. Event types:

- `PIECE_VALUE`: an UPPERCASE letter sequence with optional curly/bracket
  deletion markers and optional period-abbreviation form.
- `INSERT_OPEN` / `INSERT_CLOSE`: `(...)` immediately following a piece,
  representing container insertion `OUTER(INNER)REST`.
- `GLOSS`: `(...)` content that's clearly natural language (multi-word,
  contains spaces, etc.).
- `INDICATOR`: `[...]` content.
- `OP_WORD`: a recognised bareword: `in`, `inside`, `around`, `containing`,
  `reversed`, `rev.`, `anagram`, `hidden`, `homophone`, `sounds`, etc.
- `CHARADE_JOIN`: `+`, `,`, `then`, `next to`, `before`, `after`.
- `ASTERISK_ANAGRAM`: `*(...)` — the parenthetical is anagram fodder.
- `NOISE`: anything else (filtered).

Disambiguation rules:

- `(1-3 lowercase letters)` immediately after a piece-token = `INSERT_OPEN`
  for that letter group OR `DROP` (deletion). Distinguish by whether
  the surrounding piece has an opening uppercase: `A(POST)LE` is a
  container insertion; `CHA(p)` (where the parens come at the end) is
  a deletion.
- `(content with spaces or several letters)` after a piece-token = `GLOSS`.
- A `[...]` bracket whose content matches a phrase in the clue text
  is `INDICATOR`.

### Stage 2 — Extract pieces and operations

From the event stream:

- Each `PIECE_VALUE` produces a candidate piece with its letters.
- Each `INSERT` chain produces a container relationship: outer letters
  (the surrounding uppercase) wrap the inner (the parenthesised piece).
- Each `ASTERISK_ANAGRAM` produces an anagram piece.
- Each `OP_WORD` flags an operation. If followed by a `GLOSS` or
  `INDICATOR`, the indicator is anchored; otherwise it's a floating op.
- `CHARADE_JOIN` events delimit pieces in a charade.

### Stage 3 — Anchor pieces to clue words via DB

For each extracted piece `(value, mechanism_hint)`:

1. **Direct hit**: query `synonyms_pairs` for any clue word `w` where
   `synonym = value`. If found, anchor the piece to `w`.
2. **Abbreviation hit**: query `abbreviations` table for any clue word
   `w` where `abbreviation = value`. If found, anchor.
3. **Multi-word phrase**: try contiguous spans of clue words (2-, 3-,
   4-word) before single words.
4. **Letter-based**: if `value` equals letters of a clue word
   (`shoe → SHOE`), it's a literal/raw piece.
5. **Curly-brace remainder**: for deletion pieces with sub_kind `tail`,
   `head`, etc., the pre-deletion letters are reconstructed; look up
   that against clue words.

If no DB hit, the piece is **floating**. By-elimination from the
remaining clue words determines its source. The (source_word, value)
pair becomes a `shadow_candidate`.

### Stage 4 — Assign definition

- If `clue_definition` is set: search for that phrase in the clue text
  and tag those words as `definition`.
- Else: take the longest unaccounted span at the start or end of the
  clue (existing fallback).

### Stage 5 — By-elimination for indicators

After pieces and definition are tagged, the remaining clue words are
unaccounted. If a floating op was identified in the blog (e.g.
container, reversal), assign one or more of these residue words as the
indicator for that op.

Heuristic: pick the residue word(s) whose position in the clue makes
sense (between two pieces for a container, before/after the fodder for
reversal etc.).

### Stage 6 — Return Mapping

Same `Mapping` shape as today, plus:

- Each anchored Tag carries a `db_confirmed` flag (True if the anchor
  came from a live DB lookup).
- `shadow_candidates: list[dict]` holds the enrichment proposals.

## DB calls used

Only read-only:

```sql
-- Synonym anchor
SELECT word FROM synonyms_pairs WHERE synonym = ? COLLATE NOCASE
-- Abbreviation anchor
SELECT word FROM abbreviations WHERE abbreviation = ? COLLATE NOCASE
-- Definition lookup (sanity check on clue_definition)
SELECT 1 FROM definition_answers_augmented WHERE definition = ? AND answer = ?
```

These are all indexed lookups. For 587k clues × ~3-5 lookups per clue,
total ~2-3M queries on a fast SQLite — minutes, not hours.

## Output: shadow candidates

Each `shadow_candidate` has:

```
{
  "kind"       : "synonym" | "abbreviation" | "indicator" | "definition",
  "source_word": <clue word(s)>,
  "value"      : <token letters>,
  "subtype"    : <op subtype if indicator>,
  "evidence"   : "blog says X is a piece, by elimination it must come from Y"
}
```

Downstream: a writer process inserts these into a shadow sqlite (same
schema as `cryptic_new.db`) for use by the existing solver wrapper when
re-running clues.

## What this rebuild does NOT do

- Pure-prose blogs ("A trial takes place in a 'court', which when
  broadcast sounds like the answer") — needs LLM.
- Compound chains 3+ ops deep — basic compounds only.
- Substitution / letter-cycling / spoonerism — out of scope.
- Cryptic definitions where there's no wordplay structure — accept as
  unparseable and leave for LLM.

## What stays the same

- `schema.py`, `verifier.py`, `assembly_enumerator.py` — untouched.
- `Mapping` and `Tag` dataclass shapes — additive only.
- The HTML report, runner, etc. — unchanged.

The rebuilt mapper is a drop-in replacement for `word_role_mapper.py`.
