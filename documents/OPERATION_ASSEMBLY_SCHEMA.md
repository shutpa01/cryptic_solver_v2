# Operation / Assembly schema — the composable signature model

Status: DESIGN, agreed 2026-06-19. Target for the loader, miner, and one generic verifier
that replace the per-type engines. Supersedes the per-operation signature engines and the
14 bespoke engines (see memory/engine_audit_signature_vs_bespoke.md) by composition.

## 0. The core idea

A clue is solved by PIECES combined by an ASSEMBLY, where each piece may be transformed by
an OPERATION. Operations and assemblies are independent, composable layers:

- an OPERATION engine transforms a value (reverse, delete, anagram, select, …) and knows
  nothing about how pieces combine;
- an ASSEMBLY engine places pieces into the answer (concatenate, insert, single) and knows
  nothing about what operated on each piece;
- the SIGNATURE is the orchestration: which base feeds which operation, and which assembly
  combines the resulting pieces. It dictates what runs when.

One generic verifier reads a signature and drives the operation + assembly engines. New
clue types become new signatures (data), not new engines (code).

## 1. The answer is an atom array

The answer is N atom-slots, each `{letter, source: null}`. Solving = filling every slot's
source. This is the WFW substrate (redesign §3.2). A piece claims a contiguous SPAN of
slots; the assembly decides which span.

## 2. Pieces — contributors vs operands

A piece is a contribution from one or more clue atoms. Two kinds:

- CONTRIBUTOR — its (possibly operated) value lands in the answer; it claims a position span.
- OPERAND — consumed by an operation, never placed (the M removed from MARCH).

A piece carries: its source clue atoms, its base value (how obtained), and an optional
operation. The assembly only ever sees contributors; operands belong to their operation.

Provenance granularity is per-operation (redesign §3.4):
- per-atom: identity, reverse, deletion, selection, substitution — each answer atom has a
  definite source atom;
- span-level: anagram, homophone — the span is sourced as a whole, atoms not individually.

## 3. Operation catalogue

Each operation: `op(base_value, params, operands, target_span) -> value | None` plus a
declared provenance granularity, indicator role, and operand arity.

| kind          | base source     | indicator | operands         | params                                   | prov |
|---------------|-----------------|-----------|------------------|------------------------------------------|------|
| identity      | value           | —         | —                | —                                        | atom |
| reverse       | value           | REV_I     | —                | —                                        | atom |
| pos_delete    | value           | DEL_I     | —                | op∈{behead,curtail,outer,heartless,empty}, read from the indicator's DB sub-type | atom |
| named_delete  | value           | REM_I     | 1 (REM_F value)  | —                                        | atom |
| selection     | word            | SEL_I     | —                | rule∈{first,last,alt,middle,outer}        | atom |
| substitution  | value           | SUB_I     | 2 (from,to)      | —                                        | atom |
| anagram       | fodder atoms    | ANA_I     | —                | —                                        | span |
| homophone     | value/word      | HOM_I     | —                | —                                        | span |

Notes:
- `pos_delete` (DEL_I) and `named_delete` (REM_I) are DISTINCT operations — the role split
  the model forces. DEL_I's indicator IS the operation (position); REM_I is just a "remove"
  signpost and the operation is carried by its REM_F operand (the named value).
- `hollow` is `selection(outer)`; not a separate kind.
- anagram's base is the fodder = one OR MORE base slots whose raw letters form the pool.

## 4. Assembly catalogue

Each assembly: `assembly(contributing_pieces_with_values, answer) -> provenance | None`.

| kind      | contributors | placement                                                        |
|-----------|--------------|------------------------------------------------------------------|
| single    | 1            | the piece's value == the answer                                  |
| charade   | ≥2           | values concatenate in order == answer; piece i → its span        |
| container | 2 (outer,inner) | inner inserted into outer; outer → the two end spans, inner → the middle span |

Assemblies NEST: a contributing piece's value may itself come from a sub-assembly (container
whose inner is a charade; charade one of whose pieces is a container). The model is
recursive; the verifier resolves inner assemblies before the outer.

## 5. The signature record

Two views, both needed:
- FLAT slots — clue-order, for PLACEMENT (assign clue word-runs to slots; leftovers = links).
- NESTED structure — for VERIFICATION (which slot is whose base/operand/indicator; the
  assembly). The structure references slots by index; it is NOT in clue order.

    Signature {
      id, priority, count, origin, active,
      def_pos: 'start'|'end',
      assembly: 'single'|'charade'|'container',
      slots:   [ Slot ],          # flat, clue-order
      structure: Structure        # nested, references slot indices
    }

    Slot { index, role, n_words }
      role ∈ base:      SYN_F ABR_F LIT_F SEL_F ANA_F
            operand:    REM_F SUB_FROM SUB_TO
            indicator:  DEL_I REM_I REV_I ANA_I SEL_I SUB_I HOM_I CON_I

    Structure {
      assembly,
      pieces: [ Piece ]           # contributing pieces, in ASSEMBLY order
    }

    Piece {
      base_slots: [int],          # ≥1 slot index (≥1 only for anagram fodder)
      operation:  Operation | null,
      sub_assembly: Structure | null   # for a piece whose value is a nested assembly
    }

    Operation {
      kind,
      indicator_slot: int | null,
      operand_slots:  [int]       # REM_F for named_delete; [from,to] for substitution
    }

STORAGE: `catalog_templates` gains `assembly` (text) and `structure` (JSON) columns;
`catalog_template_slots` is unchanged (the flat slots). The 9 existing operations and the
'deletion' rows migrate by deriving a structure for each (single/charade/container + the
operation per piece).

## 6. Worked examples

ARCH — "Cunning male dropping out of parade" (def:start "Cunning")
  slots:   [0 REM_F "male"=M] [1 REM_I(2) "dropping out"] [2 SYN_F "parade"=MARCH]
  assembly: single
  structure: pieces=[ {base_slots:[2], op:{named_delete, indicator_slot:1, operand_slots:[0]}} ]
  verify:  MARCH − M = ARCH

STRAPLESS — "Small lass with pert bust lacking obvious support" (def:end)
  slots:   [0 ABR_F "Small"=S] [1 ANA_F "lass"] [2 ANA_F "pert"] [3 ANA_I "bust"]   ("with"=link)
  assembly: charade
  structure: pieces=[ {base_slots:[0], op:null},
                      {base_slots:[1,2], op:{anagram, indicator_slot:3}} ]
  verify:  S · anag(LASS PERT) = S·TRAPLESS = STRAPLESS  (S at pos1; anagram span pos2-9)

EXHORTS — "Former lover put on garment with no top, generating urges" (def:end "urges")
  assembly: charade
  structure: pieces=[ {base:"Former lover"=EX, op:null},
                      {base:"garment"=SHORTS, op:{pos_delete, indicator:"no top"→behead}} ]
  verify:  EX · behead(SHORTS) = EX·HORTS = EXHORTS   (folds in charade_deletion)

LIMESTONE — "…" container whose inner is a charade
  assembly: container; pieces=[ outer {base:"unaccompanied"=LONE},
                                inner {sub_assembly: charade(IM, EST)} ]
  verify:  LONE around (IM·EST) = LIMESTONE   (folds in container_inner_charade)

## 7. The generic verifier (replaces the per-type engines)

    solve(ctx, signature):
      for each definition split at signature.def_pos:
        place signature.slots onto clue word-runs in clue order (gaps→links last)
        for each contributing piece (deepest sub-assembly first):
            base = lookup(base_slots) by role
            value = operation.engine(base, params-from-indicator, operands, target) or base
        provenance = assembly.engine(contributing pieces' values, answer)
        if provenance complete and every clue word accounted -> Parse(pass)
    keep the best; reuse the shared scorer.

Operation engines and assembly engines are small registries keyed by `kind`; adding a kind
is a new entry, not a new solver. Placement, scoring, links-as-residue, and fail-evidence
are shared across all signatures.

## 8. Build order (staged, A/B-gated like deletion)

1. [DONE] Land the schema: `assembly` + `structure` columns; loader reads them; the generic
   verifier (core/signature_verifier.py).
2. [DONE] Migrate deletion to single-assembly + pos_delete/named_delete; IN PRODUCTION
   (cascade default). A/B: generic == deletion signature engine 3000/3000; role split applied
   (named=REM_I, positional=DEL_I).
3. [PARTIAL] Add assemblies: charade [DONE for identity pieces — generic == charade engine
   436/436 on 1500, 1 deferred SEL_F; NOT yet cascade default]; container [TODO].
4. [NEXT] The general provenance→Parse builder (identity reuse of the per-type _build does
   not cover OPERATED charade/container pieces). Then COMPOSITION: charade × deletion =
   charade_deletion, A/B vs the bespoke engine, retire it. Then operations reverse / anagram
   / selection / substitution / homophone, each A/B-gated.
5. [TODO] One verifier remains; the per-type engine files become operation/assembly registry
   entries or are deleted once their callers are gone.

## 9. Open questions

- Placement cost with nested structures: bound the DFS (cap pieces/operands per clue).
- Miner: derive the structure from each existing engine's clean pass (assembly + per-piece
  operation), the same way deletion's miner derived its recipes.
- Container outer/inner identification and multi-inner: carry as explicit piece roles.
- Provenance for nested span operations (anagram inside container): compose span- and
  atom-level entries (redesign §3.4 allows both).
- Do `single` + `identity` ever need to coexist with a definition-only clue? No — those are
  hidden/dd/cd, handled by the standalone quick checks, not this verifier.
